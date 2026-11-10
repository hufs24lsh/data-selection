"""Command-line steps of ShallowFrontier. Run `python -m shallowfrontier.cli <step> -h` for the options of each step.

  calibration  draw the calibration subset (InstructDiff's warm-up rule: random.seed(seed); random.sample(pool[:limit], k))
  features     prompt-only hidden states of the base model (GPU)
  group        label-free grouping of the features (CPU)
  score        InstructDiff scores of the pool under the base and the calibrated model (GPU); InstructDiff's own round-1
               scored.jsonl can be used instead
  noise        per-group gradient noise scale on the calibrated model, measured until the selection is settled (one GPU)
  select       shares n_g / B_g and routes B_g / b -> the selected subset
Every step refuses to overwrite an existing output.
"""

import argparse
import json
import os
import random
import sys

from .data import match_to_pool, read_jsonl, read_lines, write_lines


def _fresh(path):
    if os.path.exists(path):
        sys.exit("output exists (not overwritten): " + path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    return path


def _dump(obj, path):
    with open(_fresh(path), "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def _model(path, dtype):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, dtype)).to("cuda")


def _tokenizer(path, max_length):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, model_max_length=max_length)


def cmd_calibration(a):
    rows = read_jsonl(a.pool)
    if a.limit:
        rows = rows[:a.limit]
    rng = random.Random(a.seed)
    sub = rows if a.k >= len(rows) else rng.sample(rows, a.k)
    write_lines(_fresh(a.out), [json.dumps(r, ensure_ascii=False) for r in sub])
    print("calibration subset: %d of %d rows -> %s" % (len(sub), len(rows), a.out))


def cmd_features(a):
    import numpy as np
    from .grouping import prompt_features
    model = _model(a.model, "float16").eval()
    feats = prompt_features(model, _tokenizer(a.model, 2048), read_jsonl(a.pool), a.template)
    np.save(_fresh(a.out), feats)
    print("features %s -> %s" % (feats.shape, a.out))


def cmd_group(a):
    import numpy as np
    from .grouping import cluster
    groups, s, k, per_k = cluster(np.load(a.features))
    _dump(groups, a.out)
    print("silhouette %.3f at k %d -> %s" % (s, k, "%d groups" % len(groups) if len(groups) > 1 else "one group (no clear structure)"))
    for g, m in groups.items():
        print("  %s: %d rows" % (g, len(m)))


def cmd_score(a):
    from .scoring import score
    tok = _tokenizer(a.base, 4096)
    recs = score(_model(a.base, "float16").eval(), _model(a.calibrated, "float16").eval(), tok, read_jsonl(a.pool), a.template, a.max_length)
    kept = [r for r in recs if r is not None]
    write_lines(_fresh(a.out), [json.dumps(r, ensure_ascii=False) for r in kept])
    print("scored %d rows (%d skipped by the length rule) -> %s" % (len(kept), len(recs) - len(kept), a.out))


def cmd_noise(a):
    from .noise import measure
    pool = read_jsonl(a.pool)
    cal, missing = match_to_pool(read_jsonl(a.calibration), pool)
    if missing or not cal:
        sys.exit("calibration rows not found in the pool: %d" % missing)
    groups = json.load(open(a.groups))
    res, rounds, trace = measure(_model(a.model, "bfloat16"), _tokenizer(a.tokenizer or a.model, 4096), pool, cal, groups, a.k, a.batch,
                                 a.seed, a.max_length, a.template)
    _dump(dict(k=a.k, batch=a.batch, seed=a.seed, max_length=a.max_length, rounds=rounds, groups=res, trace=trace), a.out)
    for g, v in res.items():
        print("  %s: rows %d, B %.1f ± %.1f, B/b %.2f, share %.3f%s" % (g, v["N"], v["B"], v["se"], v["B"] / a.batch, v["share"],
                                                                     ", detection limit" if v["detection_limit"] else ""))


def cmd_select(a):
    from .selection import select
    raw = read_lines(a.pool)
    pool = [json.loads(x) for x in raw]
    rows = read_jsonl(a.scores)
    idx, missing = match_to_pool(rows, pool)
    if missing:
        print("note: %d scored rows are not in the pool" % missing)
    scores = dict(zip(idx, rows))
    groups = json.load(open(a.groups))
    noise = json.load(open(a.noise))["groups"]
    picked, info = select(groups, {g: noise[g]["B"] for g in groups}, scores, a.k, a.batch)
    write_lines(_fresh(a.out), [raw[j] for j in picked])
    _dump(dict(k=a.k, batch=a.batch, groups=info), os.path.splitext(a.out)[0] + ".summary.json")
    for g, v in info.items():
        print("  %s: B/b %.2f, share %.3f -> %d rows, %s" % (g, v["ratio"], v["share"], v["picked"], v["route"]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shallowfrontier", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    tmpl = dict(default=None, help="prompt template with {instruction} and {input} (default: the Alpaca prompt)")

    p = sub.add_parser("calibration"); p.set_defaults(fn=cmd_calibration)
    p.add_argument("--pool", required=True); p.add_argument("--k", type=int, required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None); p.add_argument("--out", required=True)

    p = sub.add_parser("features"); p.set_defaults(fn=cmd_features)
    p.add_argument("--model", required=True); p.add_argument("--pool", required=True); p.add_argument("--template", **tmpl)
    p.add_argument("--out", required=True)

    p = sub.add_parser("group"); p.set_defaults(fn=cmd_group)
    p.add_argument("--features", required=True); p.add_argument("--out", required=True)

    p = sub.add_parser("score"); p.set_defaults(fn=cmd_score)
    p.add_argument("--base", required=True); p.add_argument("--calibrated", required=True); p.add_argument("--pool", required=True)
    p.add_argument("--template", **tmpl); p.add_argument("--max-length", type=int, default=2048); p.add_argument("--out", required=True)

    p = sub.add_parser("noise"); p.set_defaults(fn=cmd_noise)
    p.add_argument("--model", required=True, help="the calibrated model"); p.add_argument("--tokenizer", default=None, help="default: --model")
    p.add_argument("--pool", required=True); p.add_argument("--calibration", required=True); p.add_argument("--groups", required=True)
    p.add_argument("--k", type=int, required=True); p.add_argument("--batch", type=int, required=True, help="the recipe's global batch b")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--max-length", type=int, default=2048); p.add_argument("--template", **tmpl)
    p.add_argument("--out", required=True)

    p = sub.add_parser("select"); p.set_defaults(fn=cmd_select)
    p.add_argument("--pool", required=True); p.add_argument("--scores", required=True); p.add_argument("--groups", required=True)
    p.add_argument("--noise", required=True); p.add_argument("--k", type=int, required=True); p.add_argument("--batch", type=int, required=True)
    p.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
