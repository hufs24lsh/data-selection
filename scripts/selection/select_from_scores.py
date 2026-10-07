import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
N = 20000
TRIM = 2000
TARGET = 2000

EXPECTED = {
    128:
        "c382b4bc65388d3ff3140e4fb3d42eafd6149d6b275974872ee30afdfec245f2",
    256:
        "61411aeffad57e5662b861a887de114b49138525f12e41ab2a39e7d555028a22",
}


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(16*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, choices=[128, 256], required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--summary", required=True)
args = ap.parse_args()

K = args.K
H = K // 2

t0 = time.perf_counter()

base = load(Path(args.base))
calib = load(Path(args.calib))
pool = load(ROOT / "data/mixed/math10k_med10k_seed42.jsonl")

if not (len(base) == len(calib) == len(pool) == N):
    raise RuntimeError("row mismatch")

L = np.asarray(
    [int(x["response_tokens_full"]) for x in base],
    dtype=np.float64,
)

dnll = np.asarray(
    [
        float(b[f"nll_{K}"]) - float(c[f"nll_{K}"])
        for b, c in zip(base, calib)
    ]
)

dh_h = np.asarray(
    [
        float(b[f"entropy_{H}"]) - float(c[f"entropy_{H}"])
        for b, c in zip(base, calib)
    ]
)

dh_k = np.asarray(
    [
        float(b[f"entropy_{K}"]) - float(c[f"entropy_{K}"])
        for b, c in zip(base, calib)
    ]
)

load_seconds = time.perf_counter() - t0

t1 = time.perf_counter()

dh = dh_k.copy()
long = L > K

dh[long] = (
    dh_k[long]
    + (1.0 - K / L[long])
    * (dh_k[long] - dh_h[long])
)

idx = list(range(N))

by_nll = sorted(
    idx,
    key=lambda i: dnll[i],
)

filtered = by_nll[
    TRIM:N-TRIM
]

selected = sorted(
    filtered,
    key=lambda i: dh[i],
)[:TARGET]

selection_seconds = time.perf_counter() - t1

out = Path(args.output)

with out.open("x", encoding="utf-8") as f:
    for i in selected:
        x = pool[i]
        f.write(
            json.dumps(
                {
                    "instruction": x["instruction"],
                    "input": x.get("input", ""),
                    "response": x["response"],
                },
                ensure_ascii=False,
            ) + "\n"
        )

actual = sha(out)

if actual != EXPECTED[K]:
    raise RuntimeError(
        f"K={K} SHA mismatch\n"
        f"actual={actual}\n"
        f"expected={EXPECTED[K]}"
    )

total_seconds = time.perf_counter() - t0

summary = {
    "K": K,
    "load_seconds": load_seconds,
    "selection_seconds": selection_seconds,
    "selector_total_seconds": total_seconds,
    "selected_sha256": actual,
}

Path(args.summary).write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(summary, indent=2))
