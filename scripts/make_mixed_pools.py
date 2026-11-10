#!/usr/bin/env python3
"""Builds the mixed math + medical pools of the paper from InstructDiff's two source files, byte for byte.

Each pool row is a source row with a "domain" field appended ("math" or "med"), serialized with json.dumps(row, ensure_ascii=False).
The row order is stored in data/mixed/<pool>.order.json (source letters and line indices only, no text); the result is checked against
the stored sha256.
  python scripts/make_mixed_pools.py                 # writes data/mixed/math10k_med10k_seed42.jsonl and math5k_med5k_seed42.jsonl
"""
import argparse
import glob
import hashlib
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [x for x in f.read().split("\n") if x]


def build(order, root=ROOT):
    """order: the parsed .order.json -> pool bytes"""
    src = {k: [json.loads(x) for x in lines(os.path.join(root, p))] for k, p in order["sources"].items()}
    out = []
    for d, i in zip(order["domains"], order["indices"]):
        out.append(json.dumps(dict(src[d][i], domain=order["domain_names"][d]), ensure_ascii=False) + "\n")
    return "".join(out).encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="repository root that holds data/")
    a = ap.parse_args()
    for f in sorted(glob.glob(os.path.join(a.root, "data", "mixed", "*.order.json"))):
        order = json.load(open(f, encoding="utf-8"))
        out = os.path.join(a.root, "data", "mixed", order["name"] + ".jsonl")
        if os.path.exists(out):
            print("exists, kept: %s" % out)
            continue
        blob = build(order, a.root)
        if hashlib.sha256(blob).hexdigest() != order["sha256"]:
            raise SystemExit("sha256 mismatch for %s: are the source files InstructDiff's originals?" % order["name"])
        with open(out, "xb") as g:
            g.write(blob)
        print("wrote %s (%d rows, sha256 verified)" % (out, len(order["indices"])))


if __name__ == "__main__":
    main()
