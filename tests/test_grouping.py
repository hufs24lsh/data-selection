import numpy as np

from shallowfrontier.grouping import cluster, kmeans, silhouette


def test_two_clear_clusters():
    g = np.random.default_rng(1)
    A = np.r_[g.normal(0, 0.05, (300, 8)) + np.eye(8)[0], g.normal(0, 0.05, (200, 8)) + np.eye(8)[1]].astype(np.float32)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    D = 1 - A @ A.T
    np.fill_diagonal(D, 0)
    a, _, ok = kmeans(A, 2)
    assert ok and len(set(a[:300].tolist())) == 1 and len(set(a[300:].tolist())) == 1 and a[0] != a[300]
    assert silhouette(D, a, 2) > 0.8
    groups, s, k, _ = cluster(A)
    assert k == 2 and sorted(len(m) for m in groups.values()) == [200, 300] and s > 0.8


def test_silhouette_matches_brute_force():
    g = np.random.default_rng(1)
    B = g.normal(size=(40, 5)).astype(np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    D = 1 - B @ B.T
    np.fill_diagonal(D, 0)
    lab = np.arange(40) % 3
    ref = []
    for i in range(40):
        ai = D[i, [j for j in range(40) if lab[j] == lab[i] and j != i]].mean()
        bi = min(D[i, [j for j in range(40) if lab[j] == c]].mean() for c in range(3) if c != lab[i])
        ref.append((bi - ai) / max(ai, bi))
    assert abs(silhouette(D, lab, 3) - float(np.mean(ref))) < 1e-5


def test_no_structure_gives_one_group():
    X = np.random.default_rng(0).normal(size=(200, 16)).astype(np.float32)
    groups, s, _, _ = cluster(X)
    assert s < 0.25 and list(groups) == ["c0"] and len(groups["c0"]) == 200
