"""Per-group gradient noise scale, measured sequentially until the selection it decides is settled.

Estimator (McCandlish et al., 2018): B = tr(Sigma) / |G|^2 from per-row gradients g_i of the summed response-token NLL on the calibration
model. Each gradient is sketched by a fixed random 0.1% of every tensor's coordinates (the ratio is unbiased under random coordinate
sampling); rows go to 16 blocks for a jackknife standard error.

Rows: each group's held-out rows (not in the calibration set), in one fixed random order, measured in rounds of b rows per group (b = the
recipe's global batch), all groups in lockstep. After each round, with s_g = (n_g/B_g) / sum_h(n_h/B_h), k_g = k s_g and r_g = se_g/B_g:
  share part  sd_g = k_g * sqrt((1 - s_g)^2 r_g^2 + sum_{h != g} s_h^2 r_h^2)        (delta method)
  route part  k_g * P(flip),  P(flip) = Phi(-|ln(B_g / b)| / r_g)                     (log-normal approximation)
A group is settled when E_g = sd_g + k_g P(flip) <= b and N_g >= B_g, or when it has no held-out rows left. A group that runs out without
a positive |G|^2 estimate gets B = N (the detection limit) and is flagged. No row count is fixed in advance.
"""

import math
import random
import zlib

SKETCH_FRACTION = 0.001
N_BLOCKS = 16
SKETCH_SEED = 20261108


def _sum_sq(x):
    """sum of squares in float64 for a torch tensor or a numpy array"""
    if hasattr(x, "double"):
        return float(x.double().pow(2).sum())
    return float((x.astype("float64") ** 2).sum())


def estimate(blk, sq, cnt):
    """blk [N_BLOCKS, d] block sums of sketched gradients, sq per-block sums of |g|^2, cnt rows per block -> (B, jackknife se)."""
    def one(S, Q, m):
        g2 = _sum_sq(S) / m / m
        tr = m / (m - 1) * (Q / m - g2)
        G = g2 - tr / m
        return tr / G if G > 0 else float("inf")
    tot, Q, m = blk.sum(0), sum(sq), sum(cnt)
    B = one(tot, Q, m)
    jk = [one(tot - blk[i], Q - sq[i], m - cnt[i]) for i in range(len(cnt)) if cnt[i]]
    mu = sum(jk) / len(jk)
    se = math.sqrt((len(jk) - 1) / len(jk) * sum((x - mu) ** 2 for x in jk)) if all(math.isfinite(x) for x in jk) else float("inf")
    return B, se


def settle(stats, sizes, k, b):
    """stats {g: dict(B, se, N, left)}, sizes {g: n_g} -> per-group decision record, or None while a group with rows left has no usable estimate."""
    use = {}
    for g, s in stats.items():
        B, se = s["B"], s["se"]
        if not (math.isfinite(B) and B > 0):
            if s["left"]:
                return None
            B, se = float(max(s["N"], 1)), float("inf")
        use[g] = (B, se)
    w = {g: sizes[g] / use[g][0] for g in use}
    tot = sum(w.values())
    sh = {g: w[g] / tot for g in w}
    rel = {g: use[g][1] / use[g][0] for g in use}
    res = {}
    for g in use:
        kg = k * sh[g]
        var = (((1 - sh[g]) * rel[g]) ** 2 if sh[g] < 1 else 0.0) + sum((sh[h] * rel[h]) ** 2 for h in use if h != g)
        sd = kg * math.sqrt(var)
        z = abs(math.log(use[g][0] / b)) / rel[g] if rel[g] > 0 else float("inf")
        pflip = 0.5 * math.erfc(z / math.sqrt(2))
        E = sd + kg * pflip
        ok = (not stats[g]["left"]) or (E <= b and stats[g]["N"] >= use[g][0])
        res[g] = dict(B=use[g][0], se=use[g][1], rel=rel[g], share=sh[g], kg=kg, sd=sd, pflip=pflip, E=E, N=stats[g]["N"], ok=ok,
                      exhausted=not stats[g]["left"], detection_limit=not (math.isfinite(stats[g]["B"]) and stats[g]["B"] > 0))
    return res


def run_rounds(names, order, sizes, k, b, add_row, estimate_of, log=print):
    """Lockstep rounds of b usable rows per group. add_row(g, j) -> bool (False: row skipped), estimate_of(g) -> (B, se, N).
    Returns (settle result, rounds, trace)."""
    pos = {g: 0 for g in names}
    rounds, trace = 0, []
    while True:
        rounds += 1
        for g in names:
            took = 0
            while took < b and pos[g] < len(order[g]):
                j = order[g][pos[g]]
                pos[g] += 1
                if add_row(g, j):
                    took += 1
        stats = {}
        for g in names:
            B, se, N = estimate_of(g)
            stats[g] = dict(B=B, se=se, N=N, left=pos[g] < len(order[g]))
        res = settle(stats, sizes, k, b)
        trace.append(dict(round=rounds, groups={g: dict(N=stats[g]["N"], B=stats[g]["B"], se=stats[g]["se"], E=(res[g]["E"] if res else None))
                                                for g in names}))
        log("round %d | %s" % (rounds, " ; ".join("%s N %d B %.1f ± %.1f E %s" % (g, stats[g]["N"], stats[g]["B"], stats[g]["se"],
                                                  ("%.0f" % res[g]["E"]) if res else "-") for g in names)))
        if res is not None and all(v["ok"] for v in res.values()):
            return res, rounds, trace
        if all(not stats[g]["left"] for g in names):
            return settle({g: dict(stats[g], left=False) for g in names}, sizes, k, b), rounds, trace


def held_out_order(groups, calibration_indices, seed):
    """Each group's rows outside the calibration set, shuffled once with random.Random(seed)."""
    order = {}
    for g in sorted(groups):
        cand = [j for j in sorted(set(groups[g])) if j not in calibration_indices]
        random.Random(seed).shuffle(cand)
        order[g] = cand
    return order


class GradientSketch:
    """Keeps a fixed random SKETCH_FRACTION of every trainable tensor's gradient coordinates in one flat buffer (post-accumulate hooks;
    every gradient is read and freed at once, so memory stays at one sketch)."""

    def __init__(self, model, fraction=SKETCH_FRACTION, seed=SKETCH_SEED):
        import torch
        self.params = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]
        self.idx, self.off, d = {}, {}, 0
        dev = next(model.parameters()).device
        for nm, p in self.params:
            kk = max(1, round(p.numel() * fraction))
            gen = torch.Generator().manual_seed((zlib.crc32(nm.encode()) ^ seed) & 0x7FFFFFFF)
            self.idx[nm] = torch.randint(p.numel(), (kk,), generator=gen).to(dev)
            self.off[nm] = (d, d + kk)
            d += kk
        self.dim, self.total = d, sum(p.numel() for _, p in self.params)
        self.buf = torch.zeros(d, device=dev, dtype=torch.float32)
        self.fired = 0
        for nm, p in self.params:
            p.register_post_accumulate_grad_hook(self._hook(nm))

    def _hook(self, nm):
        a, b = self.off[nm]
        ix = self.idx[nm]

        def hook(p):
            self.buf[a:b] = p.grad.reshape(-1)[ix].float()
            self.fired += 1
            p.grad = None
        return hook

    def reset(self):
        self.buf.zero_()
        self.fired = 0


def measure(model, tokenizer, pool, calibration_indices, groups, k, b, seed, max_length, template=None, log=print):
    """Sequential noise measurement on a calibration model loaded in bf16 on one GPU. Returns (decision per group, rounds, trace)."""
    import torch
    import torch.nn.functional as F
    from .data import prompt_of, response_of
    from .scoring import encode
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    sketch = GradientSketch(model)
    dev = sketch.buf.device
    names = sorted(groups)
    sizes = {g: len(groups[g]) for g in names}
    order = held_out_order(groups, set(calibration_indices), seed)
    st = {g: dict(blk=torch.zeros(N_BLOCKS, sketch.dim, device=dev), sq=[0.0] * N_BLOCKS, cnt=[0] * N_BLOCKS, used=0) for g in names}

    def add_row(g, j):
        e = encode(tokenizer, prompt_of(pool[j], template), response_of(pool[j]), 4096)
        if e is None:
            return False
        full, labels, pl, tl = e
        full = full[:, :max_length]
        T = min(tl, full.shape[1] - 1 - pl)
        if T <= 0:
            return False
        sketch.reset()
        logits = model(full.to(dev)).logits[0, pl:pl + T, :].float()
        F.cross_entropy(logits, labels[pl:pl + T].to(dev), reduction="sum").backward()
        if sketch.fired != len(sketch.params):
            raise RuntimeError("gradient hooks fired %d of %d" % (sketch.fired, len(sketch.params)))
        s = st[g]
        i = s["used"] % N_BLOCKS
        s["blk"][i] += sketch.buf
        s["sq"][i] += _sum_sq(sketch.buf)
        s["cnt"][i] += 1
        s["used"] += 1
        return True

    def estimate_of(g):
        s = st[g]
        if s["used"] < 2:
            return float("inf"), float("inf"), s["used"]
        B, se = estimate(s["blk"], s["sq"], s["cnt"])
        return B, se, s["used"]

    return run_rounds(names, order, sizes, k, b, add_row, estimate_of, log=log)
