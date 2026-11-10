import math

import numpy as np

from shallowfrontier.noise import estimate, held_out_order, run_rounds, settle


def test_settle_cases():
    r = settle({"a": dict(B=98.0, se=21.9, N=512, left=True)}, {"a": 10000}, 1000, 64)
    assert r["a"]["sd"] == 0 and 0 < r["a"]["pflip"] < 0.5 and r["a"]["E"] == r["a"]["kg"] * r["a"]["pflip"]
    r = settle({"a": dict(B=float("inf"), se=float("inf"), N=300, left=False), "b": dict(B=11.0, se=0.6, N=300, left=True)},
               {"a": 500, "b": 9500}, 1000, 64)
    assert r["a"]["detection_limit"] and r["a"]["B"] == 300 and r["a"]["ok"] and not math.isnan(r["b"]["E"])
    assert settle({"a": dict(B=float("inf"), se=float("inf"), N=64, left=True)}, {"a": 10}, 100, 64) is None
    r = settle({"a": dict(B=168.1, se=20.5, N=512, left=True), "b": dict(B=336.3, se=40.3, N=512, left=True)},
               {"a": 10000, "b": 10000}, 2000, 64)
    assert abs(r["a"]["share"] - 0.667) < 0.01 and r["a"]["pflip"] < 1e-6 and not r["a"]["ok"]


def _synthetic(Btrue, n, d=64, seed=0):
    g = np.random.default_rng(seed)
    G = g.normal(size=d)
    G /= np.linalg.norm(G)
    return G[None, :] + g.normal(scale=math.sqrt(Btrue / d), size=(n, d))     # tr(Sigma) = Btrue, |G| = 1


def _run(Bs, sizes, k, b, held, seed0=0):
    rows = {g: _synthetic(Bt, held, seed=seed0 + i) for i, (g, Bt) in enumerate(Bs.items())}
    st = {g: dict(blk=np.zeros((16, 64)), sq=[0.0] * 16, cnt=[0] * 16, used=0) for g in Bs}

    def add_row(g, j):
        s, x = st[g], rows[g][j]
        i = s["used"] % 16
        s["blk"][i] += x
        s["sq"][i] += float(x @ x)
        s["cnt"][i] += 1
        s["used"] += 1
        return True

    def estimate_of(g):
        s = st[g]
        if s["used"] < 2:
            return float("inf"), float("inf"), s["used"]
        B, se = estimate(s["blk"], s["sq"], s["cnt"])
        return B, se, s["used"]
    return run_rounds(sorted(Bs), {g: list(range(held)) for g in Bs}, sizes, k, b, add_row, estimate_of, log=lambda m: None)


def test_two_low_noise_groups_settle_near_truth():
    res, _, _ = _run({"m": 11.0, "d": 49.0}, {"m": 10000, "d": 10000}, 2000, 64, 4000)
    assert all(v["ok"] and not v["exhausted"] for v in res.values())
    assert abs(res["m"]["B"] / 11 - 1) < 0.35 and abs(res["d"]["B"] / 49 - 1) < 0.35


def test_wrong_route_is_rare_near_the_boundary():
    wrong = sum(_run({"x": 98.0}, {"x": 10000}, 1000, 64, 9000, seed0=100 + s)[0]["x"]["B"] < 64 for s in range(40))
    assert wrong <= 6                                            # the rule allows about b / k = 6.4% per run


def test_boundary_group_terminates():
    res, _, _ = _run({"x": 64.5}, {"x": 10000}, 1000, 64, 1500)
    assert res["x"]["ok"]


def test_held_out_order_excludes_calibration_rows_and_is_seeded():
    groups = {"c1": [5, 3, 1, 9], "c0": [0, 2, 4, 6, 8]}
    o = held_out_order(groups, {2, 3}, 42)
    assert sorted(o["c0"]) == [0, 4, 6, 8] and sorted(o["c1"]) == [1, 5, 9] and o == held_out_order(groups, {2, 3}, 42)
