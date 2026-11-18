import types

import pytest

torch = pytest.importorskip("torch")

from shallowfrontier.cli import calibrated_rows
from shallowfrontier.scoring import score
from shallowfrontier.selection import LIGHT, ORDER, routes, select


class Tok:
    """characters -> token ids in a vocabulary of 32"""

    def __call__(self, text, return_tensors="pt"):
        return types.SimpleNamespace(input_ids=torch.tensor([[ord(c) % 32 for c in text]]))


class Model:
    """fixed logits per token id; counts its forward passes"""
    device = "cpu"

    def __init__(self, seed):
        self.W = torch.randn(32, 32, generator=torch.Generator().manual_seed(seed))
        self.calls = 0

    def __call__(self, ids):
        self.calls += 1
        return types.SimpleNamespace(logits=self.W[ids])


ROWS = [dict(instruction="question %d" % i, input="", output="answer number %d" % i) for i in range(4)]
GROUPS = {"a": [0, 2], "b": [1, 3]}
NOISE = {"batch": 64, "groups": {"a": {"B": 40.0}, "b": {"B": 80.0}}}     # B/b 0.625 (light) and 1.25 (InstructDiff order)


def test_routes_boundary():
    assert routes({"a": 63.9, "b": 64.0, "c": 200.0}, 64) == {"a": LIGHT, "b": ORDER, "c": ORDER}
    assert calibrated_rows(GROUPS, NOISE) == {1, 3}


def test_calibrated_pass_only_where_the_route_reads_it():
    base = Model(0)
    full_cal, part_cal = Model(1), Model(1)
    full = score(base, full_cal, Tok(), ROWS)
    part = score(base, part_cal, Tok(), ROWS, need=calibrated_rows(GROUPS, NOISE))
    assert full_cal.calls == 4 and part_cal.calls == 2
    for i, (f, p) in enumerate(zip(full, part)):
        assert list(p) == list(f)                                         # same fields, same order
        if i in (1, 3):
            assert p == f
        else:
            assert (p["base_nll"], p["base_avg_entropy"]) == (f["base_nll"], f["base_avg_entropy"])
            assert p["instruct_nll"] is None and p["diff_nll"] is None and p["diff_entropy"] is None
    B = {g: v["B"] for g, v in NOISE["groups"].items()}
    assert select(GROUPS, B, dict(enumerate(part)), 2, 64) == select(GROUPS, B, dict(enumerate(full)), 2, 64)


def test_no_calibrated_model_needed_when_every_group_is_light():
    recs = score(Model(0), None, Tok(), ROWS, need=set())
    assert all(r["diff_entropy"] is None and r["base_avg_entropy"] is not None for r in recs)
