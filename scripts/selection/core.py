"""Pure selection logic used by the public ShallowFrontier selector.

This module is deliberately GPU-free so that input validation and ranking
semantics can be tested in CI without model checkpoints.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def _require_int_index(row: Mapping, *, source: str, position: int) -> int:
    if "index" not in row:
        raise ValueError(f"{source} row {position}: missing index")
    idx = row["index"]
    if isinstance(idx, bool) or not isinstance(idx, int):
        raise ValueError(
            f"{source} row {position}: index must be an integer, got {idx!r}"
        )
    return idx


def validate_score_rows(
    rows: Sequence[Mapping],
    *,
    source: str,
    k: int,
    expected_n: int,
) -> None:
    """Validate one scorer output without silently reordering it."""
    h = k // 2
    required_scores = (f"nll_{k}", f"entropy_{h}", f"entropy_{k}")

    if len(rows) != expected_n:
        raise ValueError(
            f"{source}: expected {expected_n} rows, got {len(rows)}"
        )

    seen: set[int] = set()
    for position, row in enumerate(rows):
        idx = _require_int_index(row, source=source, position=position)

        if not 0 <= idx < expected_n:
            raise ValueError(
                f"{source} row {position}: index {idx} outside "
                f"[0, {expected_n})"
            )
        if idx in seen:
            raise ValueError(f"{source}: duplicate index {idx}")
        seen.add(idx)

        # The frozen scorer writes rows in candidate-pool order. Requiring
        # position==index protects the downstream pool[i] lookup from a
        # same-length but permuted scorer file.
        if idx != position:
            raise ValueError(
                f"{source} row {position}: index {idx} is out of order"
            )

        if "response_tokens_full" not in row:
            raise ValueError(
                f"{source} row {position}: missing response_tokens_full"
            )
        length = row["response_tokens_full"]
        if isinstance(length, bool):
            raise ValueError(
                f"{source} row {position}: invalid response_tokens_full"
            )
        try:
            length_float = float(length)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source} row {position}: invalid response_tokens_full"
            ) from exc
        if not math.isfinite(length_float) or length_float <= 0:
            raise ValueError(
                f"{source} row {position}: response_tokens_full must be "
                "positive and finite"
            )
        if not length_float.is_integer():
            raise ValueError(
                f"{source} row {position}: response_tokens_full must be "
                "an integer-valued count"
            )

        for field in required_scores:
            if field not in row:
                raise ValueError(
                    f"{source} row {position}: missing score field {field}"
                )
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source} row {position}: non-numeric {field}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"{source} row {position}: non-finite {field}={value!r}"
                )

    expected = set(range(expected_n))
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(
            f"{source}: missing indices {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )


def validate_paired_scores(
    base: Sequence[Mapping],
    calib: Sequence[Mapping],
    *,
    k: int,
    expected_n: int,
) -> None:
    validate_score_rows(base, source="base", k=k, expected_n=expected_n)
    validate_score_rows(calib, source="calib", k=k, expected_n=expected_n)

    for position, (b, c) in enumerate(zip(base, calib)):
        if b["index"] != c["index"]:
            raise ValueError(
                f"base/calib index mismatch at row {position}: "
                f"{b['index']} != {c['index']}"
            )
        if int(b["response_tokens_full"]) != int(c["response_tokens_full"]):
            raise ValueError(
                f"base/calib response length mismatch at index {b['index']}: "
                f"{b['response_tokens_full']} != "
                f"{c['response_tokens_full']}"
            )


def asymmetric_extrapolate(
    delta_h_half: np.ndarray,
    delta_h_k: np.ndarray,
    response_lengths: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Apply the frozen asymmetric ΔH extrapolation.

    L <= K keeps the observed K-prefix value. For L > K:
        m_hat = m_K + (1 - K/L) * (m_K - m_{K/2})
    """
    half = np.asarray(delta_h_half, dtype=np.float64)
    full = np.asarray(delta_h_k, dtype=np.float64)
    lengths = np.asarray(response_lengths, dtype=np.float64)

    if not (half.shape == full.shape == lengths.shape):
        raise ValueError("delta-H and length arrays must have identical shape")
    if np.any(~np.isfinite(half)) or np.any(~np.isfinite(full)):
        raise ValueError("delta-H arrays must be finite")
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("response lengths must be positive and finite")

    corrected = full.copy()
    long = lengths > k
    corrected[long] = (
        full[long]
        + (1.0 - k / lengths[long])
        * (full[long] - half[long])
    )
    return corrected


def select_indices(
    base: Sequence[Mapping],
    calib: Sequence[Mapping],
    *,
    k: int,
    expected_n: int,
    trim: int,
    target: int,
) -> list[int]:
    """Return deterministic selected pool indices using the frozen rule."""
    if k <= 0 or k % 2:
        raise ValueError("k must be a positive even integer")
    if trim < 0 or target <= 0:
        raise ValueError("trim must be >=0 and target must be >0")
    if 2 * trim + target > expected_n:
        raise ValueError("not enough rows after trimming for target size")

    validate_paired_scores(
        base, calib, k=k, expected_n=expected_n
    )

    h = k // 2
    lengths = np.asarray(
        [int(row["response_tokens_full"]) for row in base],
        dtype=np.float64,
    )
    delta_nll = np.asarray(
        [
            float(b[f"nll_{k}"]) - float(c[f"nll_{k}"])
            for b, c in zip(base, calib)
        ],
        dtype=np.float64,
    )
    delta_h_half = np.asarray(
        [
            float(b[f"entropy_{h}"]) - float(c[f"entropy_{h}"])
            for b, c in zip(base, calib)
        ],
        dtype=np.float64,
    )
    delta_h_k = np.asarray(
        [
            float(b[f"entropy_{k}"]) - float(c[f"entropy_{k}"])
            for b, c in zip(base, calib)
        ],
        dtype=np.float64,
    )

    delta_h = asymmetric_extrapolate(
        delta_h_half, delta_h_k, lengths, k=k
    )

    indices = list(range(expected_n))

    # Explicit index tie-breakers make the deterministic behavior independent
    # of relying on Python's stable-sort implementation detail.
    by_nll = sorted(indices, key=lambda i: (delta_nll[i], i))
    filtered = by_nll[trim:expected_n - trim]
    return sorted(filtered, key=lambda i: (delta_h[i], i))[:target]
