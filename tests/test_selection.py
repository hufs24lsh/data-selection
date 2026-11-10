import pytest

from shallowfrontier.selection import LIGHT, ORDER, count_rule, select, shares


def _toy():
    scores = {j: dict(base_avg_entropy=float(40 - j), diff_nll=float(j % 7), diff_entropy=float(j)) for j in range(40)}
    groups = {"c0": list(range(20)), "c1": list(range(20, 40))}
    return scores, groups


def test_shares_and_routes():
    scores, groups = _toy()
    # B/b: c0 0.5 (light), c1 2.0 (InstructDiff order); n/B 10 vs 2.5 -> 8 and 2 of k = 10
    picked, info = select(groups, {"c0": 2.0, "c1": 8.0}, scores, 10, 4)
    assert info["c0"]["picked"] == 8 and info["c1"]["picked"] == 2
    assert info["c0"]["route"] == LIGHT and info["c1"]["route"] == ORDER
    assert picked[:8] == list(range(12, 20))                     # lowest base entropy = highest index inside c0
    assert picked[8:] == sorted(count_rule(groups["c1"], scores, 2))


def test_count_rule_trims_both_tails():
    scores = {j: dict(diff_nll=float(j), diff_entropy=float(-j)) for j in range(20)}
    # 20 rows: drop 2 at each diff_nll end (0, 1 and 18, 19), then lowest diff_entropy = highest remaining index
    assert count_rule(list(range(20)), scores, 3) == [17, 16, 15]


def test_largest_remainder_ties_keep_group_order():
    groups = {"a": list(range(3)), "b": list(range(3, 6))}
    kg, _ = shares(groups, {"a": 1.0, "b": 1.0}, 5)              # 2.5 each: the extra pick goes to the first group
    assert kg == {"a": 3, "b": 2}


def test_too_few_rows_raises():
    scores, groups = _toy()
    for j in groups["c0"]:
        scores[j]["base_avg_entropy"] = float("nan")
    with pytest.raises(ValueError):
        select(groups, {"c0": 2.0, "c1": 8.0}, scores, 10, 4)
