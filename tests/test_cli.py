import json
import random

import numpy as np

from shallowfrontier.cli import main


def _pool(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for j in range(n):
            f.write(json.dumps(dict(instruction="q%d é" % j, input="", response="a%d" % j, domain="math" if j < n // 2 else "med"), ensure_ascii=False) + "\n")


def test_calibration_matches_the_upstream_warmup_rule(tmp_path):
    _pool(tmp_path / "pool.jsonl", 50)
    main(["calibration", "--pool", str(tmp_path / "pool.jsonl"), "--k", "7", "--seed", "42", "--out", str(tmp_path / "cal.jsonl")])
    rows = [json.loads(x) for x in open(tmp_path / "pool.jsonl", encoding="utf-8")]
    random.seed(42)                                               # InstructDiff: random.seed(seed); random.sample(data, max_samples)
    want = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in random.sample(rows, 7))
    assert open(tmp_path / "cal.jsonl", encoding="utf-8").read() == want


def test_group_and_select_end_to_end(tmp_path):
    n = 40
    _pool(tmp_path / "pool.jsonl", n)
    g = np.random.default_rng(0)
    feats = np.r_[g.normal(0, .05, (20, 8)) + np.eye(8)[0], g.normal(0, .05, (20, 8)) + np.eye(8)[1]].astype(np.float16)
    np.save(tmp_path / "f.npy", feats)
    main(["group", "--features", str(tmp_path / "f.npy"), "--out", str(tmp_path / "groups.json")])
    groups = json.load(open(tmp_path / "groups.json"))
    assert sorted(len(m) for m in groups.values()) == [20, 20]
    with open(tmp_path / "scored.jsonl", "w") as f:
        for j in range(n):
            f.write(json.dumps(dict(instruction="q%d é" % j, input="", response="a%d" % j, base_avg_entropy=float(n - j),
                                    diff_nll=float(j % 7), diff_entropy=float(j))) + "\n")
    json.dump(dict(groups={gname: dict(B=2.0 if gname == "c0" else 8.0) for gname in groups}), open(tmp_path / "noise.json", "w"))
    main(["select", "--pool", str(tmp_path / "pool.jsonl"), "--scores", str(tmp_path / "scored.jsonl"), "--groups", str(tmp_path / "groups.json"),
          "--noise", str(tmp_path / "noise.json"), "--k", "10", "--batch", "4", "--out", str(tmp_path / "sel.jsonl")])
    lines = open(tmp_path / "sel.jsonl", encoding="utf-8").read().splitlines()
    summary = json.load(open(tmp_path / "sel.summary.json"))
    assert len(lines) == 10 and sum(v["picked"] for v in summary["groups"].values()) == 10
    raw = open(tmp_path / "pool.jsonl", encoding="utf-8").read().splitlines()
    assert all(x in raw for x in lines)                          # rows are written back byte for byte
