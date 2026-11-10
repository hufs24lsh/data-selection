"""Label-free grouping.

Features: for every pool row, the base model's last-layer hidden state averaged over the prompt tokens (the response is never read;
prompts are cut at 512 tokens). Groups: cosine k-means with k-means++ seeding, k searched from 2 to floor(sqrt(n / 2)), the k with the
largest exact mean silhouette wins; if that silhouette is below 0.25 ("no substantial structure", Kaufman & Rousseeuw) the pool is one group.
"""

import math

import numpy as np

from .data import prompt_of

SILHOUETTE_FLOOR = 0.25


def kmeans(X, k, seed=0):
    """Cosine k-means on unit rows X with k-means++ seeding; runs until no assignment changes. Returns (labels, iterations, converged)."""
    g = np.random.default_rng(seed)
    C = [X[g.integers(len(X))]]
    for _ in range(1, k):
        d = np.maximum(np.min(1 - X @ np.array(C).T, axis=1), 0)
        C.append(X[g.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    a = np.argmax(X @ C.T, axis=1)
    for it in range(1, len(X) + 1):
        C = np.array([X[a == c].mean(0) if np.any(a == c) else C[c] for c in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        b = np.argmax(X @ C.T, axis=1)
        if np.array_equal(a, b):
            return a, it, True
        a = b
    return a, len(X), False


def silhouette(D, a, k):
    """Exact mean silhouette from the full distance matrix D (zero diagonal); singleton clusters score 0."""
    onehot = np.zeros((len(a), k), dtype=np.float32)
    onehot[np.arange(len(a)), a] = 1
    M = D @ onehot
    cnt = onehot.sum(0)
    own = cnt[a]
    ai = np.where(own > 1, M[np.arange(len(a)), a] / np.maximum(own - 1, 1), 0)
    other = np.where((np.arange(k)[None, :] != a[:, None]) & (cnt[None, :] > 0), M / np.maximum(cnt[None, :], 1), np.inf)
    bi = other.min(1)
    s = np.where((own > 1) & np.isfinite(bi), (bi - ai) / np.maximum(np.maximum(ai, bi), 1e-12), 0)
    return float(s.mean())


def cluster(features, seed=0, floor=SILHOUETTE_FLOOR):
    """features [n, dim] -> (groups {"c<i>": [row indices]}, best silhouette, chosen k, silhouette per k)."""
    X = np.asarray(features, dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    D = 1 - X @ X.T
    np.fill_diagonal(D, 0)
    kmax = int(math.floor(math.sqrt(len(X) / 2)))
    per_k, best = {}, (-2.0, 1, None)
    for k in range(2, kmax + 1):
        a, it, ok = kmeans(X, k, seed)
        s = silhouette(D, a, k)
        per_k[k] = dict(silhouette=s, iterations=it, converged=ok)
        if s > best[0]:
            best = (s, k, a)
    s, k, a = best
    if a is None or s < floor:
        a = np.zeros(len(X), dtype=int)
    groups = {"c%d" % c: [int(j) for j in np.where(a == c)[0]] for c in sorted(set(a.tolist()))}
    return groups, s, k, per_k


def prompt_features(model, tokenizer, rows, template=None, max_prompt_tokens=512, batch_size=32, device="cuda"):
    """Last-layer hidden states averaged over each row's prompt tokens, float16 [n, hidden]. Batches are sorted by length."""
    import torch
    ids = [tokenizer(prompt_of(r, template), add_special_tokens=False)["input_ids"][:max_prompt_tokens] for r in rows]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = None
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            m = max(len(ids[i]) for i in b)
            x = torch.zeros((len(b), m), dtype=torch.long)
            att = torch.zeros((len(b), m), dtype=torch.long)
            for r, i in enumerate(b):
                x[r, :len(ids[i])] = torch.tensor(ids[i])
                att[r, :len(ids[i])] = 1
            h = model(input_ids=x.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states[-1].float()
            v = (h * att.to(device)[..., None]).sum(1) / att.to(device).sum(1, keepdim=True)
            if out is None:
                out = np.zeros((len(rows), v.shape[1]), dtype=np.float16)
            out[b] = v.cpu().numpy().astype(np.float16)
    if out is None or not np.isfinite(out.astype(np.float32)).all():
        raise ValueError("non-finite or empty prompt features")
    return out
