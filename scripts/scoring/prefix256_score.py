import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

DEPTHS = (32, 64, 128, 256)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"JSON parse failure {path} line {lineno}: {e}"
                ) from e
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    output_path = Path(args.output)
    meta_path = output_path.with_suffix(".meta.json")

    if output_path.exists() or meta_path.exists():
        raise RuntimeError("refusing to overwrite output")

    data = load_jsonl(data_path)

    if len(data) != 20000:
        raise RuntimeError(f"expected 20000 rows, got {len(data)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        model_max_length=args.max_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    device = model.device
    model.eval()

    t0 = time.perf_counter()

    skipped = 0
    total_full_forward_tokens = 0
    total_forward_tokens = {k: 0 for k in DEPTHS}
    total_response_tokens = 0

    with output_path.open("x", encoding="utf-8") as fout:
        with torch.inference_mode():
            for idx, item in enumerate(data):
                instruction = item["instruction"]
                user_input = item.get("input", "")
                response = item["response"]

                prompt = DEFAULT_PROMPT.format_map(
                    {
                        "instruction": instruction,
                        "input": user_input,
                    }
                )

                # Match authoritative scorer's tokenization semantics.
                prompt_ids_cpu = tokenizer(
                    prompt,
                    return_tensors="pt",
                ).input_ids

                response_ids_cpu = tokenizer(
                    response,
                    return_tensors="pt",
                ).input_ids

                if prompt_ids_cpu.shape[1] > args.max_length:
                    skipped += 1
                    continue

                if response_ids_cpu.shape[1] > args.max_length:
                    skipped += 1
                    continue

                full_ids_cpu = tokenizer(
                    prompt + "\n" + response,
                    return_tensors="pt",
                ).input_ids

                prompt_raw_len = int(prompt_ids_cpu.shape[1])

                # Exact authoritative response-region length:
                # labels length = F-1
                # prompt_len = P-1
                # target_len = (F-1) - (P-1) = F-P
                full_len = int(full_ids_cpu.shape[1])
                target_len = full_len - prompt_raw_len

                if target_len <= 0:
                    raise RuntimeError(
                        f"non-positive response region at row {idx}"
                    )

                # Research S2 performs one forward through depth <=256.
                used_target_len = min(256, target_len)
                capped_len = prompt_raw_len + used_target_len

                input_ids = full_ids_cpu[:, :capped_len].to(device)

                logits = model(input_ids).logits[:, :-1, :]

                # Same shifted region used by authoritative scorer.
                start = prompt_raw_len - 1
                target_logits = logits[
                    :,
                    start:start + used_target_len,
                    :,
                ]

                target_labels = input_ids[
                    :,
                    prompt_raw_len:
                    prompt_raw_len + used_target_len,
                ]

                if target_logits.shape[1] != used_target_len:
                    raise RuntimeError(
                        f"logit length mismatch row={idx}: "
                        f"{target_logits.shape[1]} vs {used_target_len}"
                    )

                log_probs = F.log_softmax(target_logits, dim=-1)

                token_log_probs = log_probs.gather(
                    2,
                    target_labels.unsqueeze(2),
                ).squeeze(2)

                token_nll = -token_log_probs.squeeze(0)

                probs = F.softmax(target_logits, dim=-1)

                token_entropy = -torch.sum(
                    probs * log_probs,
                    dim=-1,
                ).squeeze(0)

                row = {
                    "index": idx,
                    "role": args.role,
                    "prompt_input_tokens": prompt_raw_len,
                    "response_tokens_full": target_len,
                    "full_forward_tokens": full_len,
                }

                total_full_forward_tokens += full_len
                total_response_tokens += target_len

                for k in DEPTHS:
                    d = min(k, target_len)

                    # used_target_len is >= d for every requested k.
                    row[f"nll_{k}"] = float(
                        token_nll[:d].mean().item()
                    )

                    row[f"entropy_{k}"] = float(
                        token_entropy[:d].mean().item()
                    )

                    forward_tokens_k = prompt_raw_len + d
                    row[f"forward_tokens_{k}"] = forward_tokens_k
                    total_forward_tokens[k] += forward_tokens_k

                fout.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

                if (idx + 1) % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"{args.role}: {idx + 1}/20000 "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

                # Drop large vocab-sized tensors before next example.
                del (
                    input_ids,
                    logits,
                    target_logits,
                    target_labels,
                    log_probs,
                    token_log_probs,
                    token_nll,
                    probs,
                    token_entropy,
                )

    wall = time.perf_counter() - t0

    if skipped != 0:
        raise RuntimeError(
            f"authoritative scorer compatibility violation: skipped={skipped}"
        )

    meta = {
        "status": "PASS",
        "role": args.role,
        "model": str(model_path),
        "rows": len(data),
        "skipped": skipped,
        "depths": list(DEPTHS),
        "wall_seconds": wall,
        "examples_per_second": len(data) / wall,
        "total_full_forward_tokens_one_model":
            total_full_forward_tokens,
        "total_response_tokens_full":
            total_response_tokens,
        "total_forward_tokens_one_model_by_depth":
            {
                str(k): total_forward_tokens[k]
                for k in DEPTHS
            },
        "output_sha256": sha256(output_path),
    }

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(f"S2 {args.role.upper()} COMPLETE")
    print("=" * 78)
    print(json.dumps(meta, indent=2))
    print("meta_sha256 =", sha256(meta_path))


if __name__ == "__main__":
    main()
