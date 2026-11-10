#!/usr/bin/env python3
"""Medical multiple-choice evaluation (MedQA, MMLU-medical, MedMCQA) with vLLM, one GPU.

  --protocol mcf       (default, used in the paper) multiple-choice formulation: only the option letters compete. Every vocabulary
                       token that decodes, stripped and upper-cased, to A/B/C/D counts for that letter; a letter's score is the
                       logsumexp of its tokens' log-probabilities (as in OLMES-style MCF scoring).
  --protocol released  the released InstructDiff script's rule, for comparison: argmax over A-D among the first-token top-10
                       log-probabilities, where a later (lower-probability) variant of the same letter overwrites an earlier one;
                       else the first letter in the generated token; else 'A'. See docs/evaluation.md.

Prompts are those of the released InstructDiff medical evaluation script.
  python eval/medeval/mcf_eval.py --model runs/final --out results/final_med_mcf.json
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

OPTIONS = ("A", "B", "C", "D")
DATASETS = ("medqa", "mmlu", "medmcqa")
FILES = dict(medqa="medqa_test.jsonl", mmlu="mmlu_medical_test.jsonl", medmcqa="medmcqa_test.jsonl")
HERE = os.path.dirname(os.path.abspath(__file__))


def format_question_medqa(item):
    formatted = f"Question: {item['question']}\n\nOptions:\n"
    for key, value in item["options"].items():
        formatted += f"{key}. {value}\n"
    return formatted + "\nAnswer:"


def format_question_mmlu(item):
    formatted = f"Question: {item['question']}\n\nOptions:\n"
    for i, choice in enumerate(item["choices"]):
        formatted += f"{chr(65 + i)}. {choice}\n"
    return formatted + "\nAnswer:"


def format_question_medmcqa(item):
    formatted = f"Question: {item['question']}\n\nOptions:\n"
    for i, option in enumerate([item["opa"], item["opb"], item["opc"], item["opd"]]):
        formatted += f"{chr(65 + i)}. {option}\n"
    return formatted + "\nAnswer:"


FORMAT = dict(medqa=format_question_medqa, mmlu=format_question_mmlu, medmcqa=format_question_medmcqa)


def gold(item, dataset):
    if dataset == "medqa":
        return item["answer_idx"]
    if dataset == "mmlu":
        return chr(65 + item["answer"])
    return chr(65 + item["cop"]) if item["cop"] != -1 else "A"


def released_prediction(top, text):
    """top: {decoded token: logprob} of the first position, in the engine's order (highest first)."""
    option = {}
    for tok, lp in top.items():
        t = tok.strip().upper()
        if t in OPTIONS:
            option[t] = lp                              # the released loop keeps the last token that maps to a letter
    if option:
        return max(option, key=option.get), "top10_option"
    resp = text.strip().upper()
    for ch in OPTIONS:
        if ch in resp:
            return ch, "text_fallback"
    return "A", "default_A"


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def mcf_prediction(lp_by_id, option_token_ids):
    scores = {}
    for opt, ids in option_token_ids.items():
        vals = [lp_by_id[i] for i in ids if i in lp_by_id]
        if not vals:
            raise RuntimeError("no returned logprob for option %s" % opt)
        scores[opt] = logsumexp(vals)
    return max(scores, key=scores.get), scores


def option_tokens(tokenizer):
    groups = defaultdict(list)
    for token_id in range(len(tokenizer)):
        norm = tokenizer.decode([token_id]).strip().upper()
        if norm in OPTIONS:
            groups[norm].append(token_id)
    missing = [o for o in OPTIONS if not groups[o]]
    if missing:
        raise RuntimeError("no single-token representation for %s" % missing)
    return dict(groups)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="mcf", choices=["mcf", "released"])
    ap.add_argument("--test-dir", default=os.path.join(HERE, "test_data"))
    ap.add_argument("--save-items", action="store_true", help="also store every item's prediction")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("output exists (not overwritten): " + a.out)
    from vllm import LLM, SamplingParams
    t0 = time.time()
    if a.protocol == "released":
        llm = LLM(model=a.model, tensor_parallel_size=1)
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=10)
        ids = None
    else:
        # logprobs of the masked distribution, so every allowed token is returned (the mask only renormalizes)
        llm = LLM(model=a.model, tensor_parallel_size=1, max_logprobs=512, logprobs_mode="processed_logprobs")
        ids = option_tokens(llm.get_tokenizer())
        allowed = sorted({i for v in ids.values() for i in v})
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=len(allowed), allowed_token_ids=allowed)
    tok = llm.get_tokenizer()
    out = dict(protocol=a.protocol, model=a.model, datasets={})
    for ds in DATASETS:
        with open(os.path.join(a.test_dir, FILES[ds]), encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        recs = []
        for x, o in zip(data, llm.generate([FORMAT[ds](x) for x in data], sp)):
            first = o.outputs[0]
            if a.protocol == "released":
                top = {tok.decode([t]): float(lp.logprob) for t, lp in (first.logprobs[0].items() if first.logprobs else [])}
                pred, mode = released_prediction(top, first.text)
            else:
                pred, _ = mcf_prediction({t: float(v.logprob) for t, v in first.logprobs[0].items()}, ids)
                mode = "mcf"
            recs.append(dict(gold=gold(x, ds), pred=pred, mode=mode))
        acc = sum(r["pred"] == r["gold"] for r in recs) / len(recs)
        out["datasets"][ds] = dict(n=len(recs), accuracy=acc, modes=dict(Counter(r["mode"] for r in recs)), **(dict(items=recs) if a.save_items else {}))
        print("%-8s n %d · accuracy %.4f" % (ds, len(recs), acc), flush=True)
    out["average"] = sum(out["datasets"][d]["accuracy"] for d in DATASETS) / len(DATASETS)
    out["seconds"] = time.time() - t0
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("average %.4f (%s) -> %s" % (out["average"], a.protocol, a.out))


if __name__ == "__main__":
    main()
