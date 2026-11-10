"""Group shares and routes.

For groups g with sizes n_g and measured gradient noise scales B_g, and a recipe batch size b:
  share_g  proportional to n_g / B_g, rounded to integers with the largest-remainder rule (the k picks always add up);
  route_g  B_g / b < 1 -> the group's rows with the lowest base-model entropy ("light");
           otherwise   -> InstructDiff's count rule inside the group: drop 10% at each tail of ΔNLL, keep the lowest ΔH.
Scores follow InstructDiff's scored.jsonl schema: base_avg_entropy, diff_nll, diff_entropy (see shallowfrontier.scoring).
"""

import math

LIGHT, ORDER = "light (lowest base entropy)", "InstructDiff order"


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def count_rule(candidates, scores, k, trim=0.1):
    """InstructDiff's count rule on a candidate list: sort by diff_nll, drop int(n * trim) rows at each end, then take the k lowest diff_entropy."""
    c = [j for j in candidates if finite(scores[j]["diff_nll"]) and finite(scores[j]["diff_entropy"])]
    order = sorted(c, key=lambda j: scores[j]["diff_nll"])
    n = len(order)
    t = int(n * trim)
    if 2 * t >= n:
        t = 0
    return sorted(order[t:n - t], key=lambda j: scores[j]["diff_entropy"])[:k]


def shares(groups, B, k):
    """Integer picks per group, proportional to n_g / B_g (largest remainder; ties keep the order of `groups`)."""
    w = {g: len(m) / B[g] for g, m in groups.items()}
    tot = sum(w.values())
    kg = {g: int(math.floor(k * w[g] / tot)) for g in groups}
    for g in sorted(groups, key=lambda g: -(k * w[g] / tot - kg[g]))[:k - sum(kg.values())]:
        kg[g] += 1
    return kg, {g: w[g] / tot for g in groups}


def select(groups, B, scores, k, b):
    """groups {name: [pool indices]}, B {name: noise scale}, scores {pool index: scored record} -> (sorted pool indices, per-group info)."""
    kg, share = shares(groups, B, k)
    picked, info = [], {}
    for g, members in groups.items():
        ratio = B[g] / b
        if ratio < 1:
            pick = sorted((j for j in members if finite(scores[j].get("base_avg_entropy"))),
                          key=lambda j: (scores[j]["base_avg_entropy"], j))[:kg[g]]
            route = LIGHT
        else:
            pick = count_rule(members, scores, kg[g])
            route = ORDER
        if len(pick) != kg[g]:
            raise ValueError("group %s: %d usable rows for %d picks" % (g, len(pick), kg[g]))
        picked += pick
        info[g] = dict(rows=len(members), B=B[g], ratio=ratio, share=share[g], picked=kg[g], route=route)
    picked = sorted(set(picked))
    if len(picked) != k:
        raise ValueError("selected %d rows, expected %d" % (len(picked), k))
    return picked, info
