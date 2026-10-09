#!/usr/bin/env python3
"""Build non-frozen correction-ablation selections from saved score files.

This script is for new research validation. It does not replace or rewrite the
frozen K128/K256 artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path

from core import select_indices_variant

ROOT = Path.cwd()
N = 20000
TRIM = 2000
TARGET = 2000

MODES = {
    "raw": (False, False),
    "nll-only": (True, False),
    "h-only": (False, True),
    "both": (True, True),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, choices=[128, 256], required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.summary.exists():
        raise RuntimeError("refusing to overwrite ablation output")

    base = load_jsonl(args.base)
    calib = load_jsonl(args.calib)
    pool = load_jsonl(args.pool)

    if len(pool) != N:
        raise RuntimeError(f"expected {N} pool rows, got {len(pool)}")

    correct_nll, correct_h = MODES[args.mode]
    selected = select_indices_variant(
        base,
        calib,
        k=args.K,
        expected_n=N,
        trim=TRIM,
        target=TARGET,
        correct_nll=correct_nll,
        correct_h=correct_h,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as f:
        for i in selected:
            item = pool[i]
            f.write(
                json.dumps(
                    {
                        "instruction": item["instruction"],
                        "input": item.get("input", ""),
                        "response": item["response"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "status": "EXPERIMENTAL_UNFROZEN",
        "K": args.K,
        "mode": args.mode,
        "correct_nll": correct_nll,
        "correct_h": correct_h,
        "selected_rows": len(selected),
        "selected_indices": selected,
        "selected_sha256": sha256(args.output),
        "base_scores_sha256": sha256(args.base),
        "calib_scores_sha256": sha256(args.calib),
        "pool_sha256": sha256(args.pool),
    }

    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
