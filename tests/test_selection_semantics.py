import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "selection" / "core.py"

spec = importlib.util.spec_from_file_location("selection_core", CORE_PATH)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)


def make_rows(
    n,
    *,
    k=128,
    nll=None,
    dh_half=None,
    dh_k=None,
    lengths=None,
):
    h = k // 2
    nll = list(range(n)) if nll is None else list(nll)
    dh_half = [0.0] * n if dh_half is None else list(dh_half)
    dh_k = [0.0] * n if dh_k is None else list(dh_k)
    lengths = [k] * n if lengths is None else list(lengths)

    base = []
    calib = []
    for i in range(n):
        base.append(
            {
                "index": i,
                "response_tokens_full": lengths[i],
                f"nll_{k}": float(nll[i]),
                f"entropy_{h}": float(dh_half[i]),
                f"entropy_{k}": float(dh_k[i]),
            }
        )
        calib.append(
            {
                "index": i,
                "response_tokens_full": lengths[i],
                f"nll_{k}": 0.0,
                f"entropy_{h}": 0.0,
                f"entropy_{k}": 0.0,
            }
        )
    return base, calib


def test_prefix_half_and_k_entropy_are_distinct_inputs():
    # Synthetic prefix means: first K/2 tokens have mean 1; first K have
    # mean 2. This guards the intended use of separate K/2 and K statistics.
    k = 4
    token_entropy = np.asarray([1.0, 1.0, 3.0, 3.0])
    entropy_half = float(token_entropy[: k // 2].mean())
    entropy_k = float(token_entropy[:k].mean())

    assert entropy_half == pytest.approx(1.0)
    assert entropy_k == pytest.approx(2.0)

    corrected = core.asymmetric_extrapolate(
        np.asarray([entropy_half]),
        np.asarray([entropy_k]),
        np.asarray([8.0]),
        k=k,
    )
    assert corrected[0] == pytest.approx(2.5)


def test_asymmetric_correction_keeps_short_responses_unchanged():
    got = core.asymmetric_extrapolate(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([4.0, 5.0, 6.0]),
        np.asarray([64.0, 128.0, 129.0]),
        k=128,
    )

    assert got[0] == pytest.approx(4.0)
    assert got[1] == pytest.approx(5.0)
    expected_long = 6.0 + (1.0 - 128.0 / 129.0) * (6.0 - 3.0)
    assert got[2] == pytest.approx(expected_long)


def test_asymmetric_correction_matches_closed_form_for_long_response():
    got = core.asymmetric_extrapolate(
        np.asarray([2.0]),
        np.asarray([5.0]),
        np.asarray([256.0]),
        k=128,
    )
    assert got[0] == pytest.approx(6.5)


def test_q10_q90_trimming_then_delta_h_ranking():
    n = 10
    base, calib = make_rows(
        n,
        nll=list(range(n)),
        dh_k=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    )

    selected = core.select_indices(
        base,
        calib,
        k=128,
        expected_n=n,
        trim=1,
        target=3,
    )

    # NLL ranks 0 and 9 are removed. Among 1..8, the lowest ΔH values
    # belong to indices 8, 7, 6.
    assert selected == [8, 7, 6]


def test_ties_use_index_as_explicit_deterministic_tiebreaker():
    base, calib = make_rows(
        8,
        nll=[0.0] * 8,
        dh_k=[0.0] * 8,
    )

    first = core.select_indices(
        base, calib, k=128, expected_n=8, trim=1, target=4
    )
    second = core.select_indices(
        base, calib, k=128, expected_n=8, trim=1, target=4
    )

    assert first == [1, 2, 3, 4]
    assert second == first


def test_exact_2000_selected_after_decile_trimming():
    n = 2500
    base, calib = make_rows(
        n,
        nll=list(range(n)),
        dh_k=list(reversed(range(n))),
    )

    selected = core.select_indices(
        base,
        calib,
        k=128,
        expected_n=n,
        trim=250,
        target=2000,
    )

    assert len(selected) == 2000
    assert len(set(selected)) == 2000
    assert min(selected) >= 250
    assert max(selected) <= 2249


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda b, c: b.__setitem__(
                1, {**b[1], "index": 0}
            ),
            "duplicate index",
        ),
        (
            lambda b, c: b.__setitem__(
                1, {**b[1], "index": 4}
            ),
            "outside",
        ),
        (
            lambda b, c: b.__setitem__(
                0, {k: v for k, v in b[0].items() if k != "index"}
            ),
            "missing index",
        ),
        (
            lambda b, c: b.__setitem__(
                0, {k: v for k, v in b[0].items() if k != "nll_128"}
            ),
            "missing score field",
        ),
        (
            lambda b, c: b.__setitem__(
                0, {**b[0], "nll_128": float("nan")}
            ),
            "non-finite",
        ),
        (
            lambda b, c: b.__setitem__(
                0, {**b[0], "entropy_64": float("inf")}
            ),
            "non-finite",
        ),
    ],
)
def test_invalid_score_schema_fails_loudly(mutator, message):
    base, calib = make_rows(4)
    mutator(base, calib)

    with pytest.raises(ValueError, match=message):
        core.select_indices(
            base,
            calib,
            k=128,
            expected_n=4,
            trim=0,
            target=2,
        )


def test_permuted_rows_are_rejected_even_when_lengths_match():
    base, calib = make_rows(4)
    base[0], base[1] = base[1], base[0]

    with pytest.raises(ValueError, match="out of order"):
        core.select_indices(
            base,
            calib,
            k=128,
            expected_n=4,
            trim=0,
            target=2,
        )


def test_base_calibration_response_lengths_must_match():
    base, calib = make_rows(4)
    calib[2]["response_tokens_full"] = 127

    with pytest.raises(ValueError, match="response length mismatch"):
        core.select_indices(
            base,
            calib,
            k=128,
            expected_n=4,
            trim=0,
            target=2,
        )


def test_target_and_trim_boundary_is_validated():
    base, calib = make_rows(4)

    with pytest.raises(ValueError, match="not enough rows"):
        core.select_indices(
            base,
            calib,
            k=128,
            expected_n=4,
            trim=2,
            target=1,
        )
