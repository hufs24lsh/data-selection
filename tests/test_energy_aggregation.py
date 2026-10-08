import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENERGY_PATH = ROOT / "scripts" / "energy" / "aggregate_energy.py"

spec = importlib.util.spec_from_file_location("aggregate_energy", ENERGY_PATH)
energy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(energy)


def test_energy_reaggregation_from_synthetic_counter_log(tmp_path):
    csv_path = tmp_path / "energy.csv"
    marker_path = tmp_path / "markers.csv"

    rows = [
        "timestamp_utc,gpu0_energy_mJ,gpu1_energy_mJ",
    ]
    for sec in range(11):
        rows.append(
            f"2026-01-01T00:00:{sec:02d}+00:00,"
            f"{100000 + 1000 * sec},"
            f"{200000 + 2000 * sec}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    marker_path.write_text(
        "2026-01-01T00:00:02.200+00:00,SCORING,START\n"
        "2026-01-01T00:00:07.800+00:00,SCORING,END\n",
        encoding="utf-8",
    )

    result = energy.aggregate(
        csv_path,
        marker_path,
        max_boundary_gap_s=0.5,
        idle_power_w=30.0,
    )

    assert result["scope"] == "GPU_ONLY"
    assert len(result["stages"]) == 1

    stage = result["stages"][0]
    assert stage["stage"] == "SCORING"
    # Nearest samples are t=2 and t=8.
    assert stage["gpu0_energy_mJ"] == 6000
    assert stage["gpu1_energy_mJ"] == 12000
    assert stage["gross_energy_mJ"] == 18000
    assert stage["gross_gpu_kWh"] == pytest.approx(18000 / 3_600_000_000.0)
    assert stage["marker_duration_seconds"] == pytest.approx(5.6)
    assert stage["start_boundary_gap_seconds"] == pytest.approx(0.2)
    assert stage["end_boundary_gap_seconds"] == pytest.approx(0.2)

    expected_idle = 30.0 * 5.6 / 3_600_000.0
    assert stage["idle_energy_kWh"] == pytest.approx(expected_idle)
    assert stage["idle_adjusted_gpu_kWh"] == pytest.approx(
        stage["gross_gpu_kWh"] - expected_idle
    )


def test_utc_z_timestamps_in_raw_log_and_stage_markers(tmp_path):
    assert energy.parse_time("2026-01-01T00:00:00Z") == (
        energy.parse_time("2026-01-01T00:00:00+00:00")
    )

    energy_csv = tmp_path / "energy_z.csv"
    marker_csv = tmp_path / "markers_z.csv"

    energy_csv.write_text(
        "timestamp_utc,gpu0_energy_mJ,gpu1_energy_mJ\n"
        "2026-01-01T00:00:00Z,100,200\n"
        "2026-01-01T00:00:01Z,110,220\n"
        "2026-01-01T00:00:02Z,130,240\n",
        encoding="utf-8",
    )

    marker_csv.write_text(
        "2026-01-01T00:00:00.100Z,S,START\n2026-01-01T00:00:01.900Z,S,END\n",
        encoding="utf-8",
    )

    result = energy.aggregate(energy_csv, marker_csv, max_boundary_gap_s=0.2)
    stage = result["stages"][0]

    assert stage["gpu0_energy_mJ"] == 30
    assert stage["gpu1_energy_mJ"] == 40
    assert stage["gross_energy_mJ"] == 70


def test_linear_boundary_policy(tmp_path):
    energy_csv = tmp_path / "energy_linear.csv"
    marker_csv = tmp_path / "markers_linear.csv"

    energy_csv.write_text(
        "timestamp_utc,gpu0_energy_mJ,gpu1_energy_mJ\n"
        "2026-01-01T00:00:00Z,0,0\n"
        "2026-01-01T00:00:01Z,100,200\n"
        "2026-01-01T00:00:02Z,400,800\n",
        encoding="utf-8",
    )

    marker_csv.write_text(
        "2026-01-01T00:00:00.250Z,S,START\n2026-01-01T00:00:01.750Z,S,END\n",
        encoding="utf-8",
    )

    linear = energy.aggregate(
        energy_csv,
        marker_csv,
        boundary_policy="linear",
        idle_power_w=46.953,
    )

    nearest = energy.aggregate(
        energy_csv,
        marker_csv,
    )

    stage = linear["stages"][0]

    assert linear["boundary_policy"] == "linear"
    assert nearest["boundary_policy"] == "nearest"

    assert stage["gpu0_energy_mJ"] == pytest.approx(300)
    assert stage["gpu1_energy_mJ"] == pytest.approx(600)
    assert stage["gross_energy_mJ"] == pytest.approx(900)
    assert stage["gross_gpu_kWh"] == pytest.approx(900 / 3_600_000_000.0)

    assert nearest["stages"][0]["gross_energy_mJ"] == 1200

    assert stage["marker_duration_seconds"] == pytest.approx(1.5)
    assert stage["idle_energy_kWh"] == pytest.approx(46.953 * 1.5 / 3_600_000.0)

    assert "counter_start_sample_utc" not in stage
    assert stage["counter_start_bracket_utc"] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
    ]


def test_energy_reaggregation_rejects_large_marker_gap(tmp_path):
    csv_path = tmp_path / "energy.csv"
    marker_path = tmp_path / "markers.csv"

    csv_path.write_text(
        "timestamp_utc,gpu0_energy_mJ,gpu1_energy_mJ\n"
        "2026-01-01T00:00:00+00:00,1,1\n"
        "2026-01-01T00:00:10+00:00,2,2\n",
        encoding="utf-8",
    )
    marker_path.write_text(
        "2026-01-01T00:00:04+00:00,S,START\n2026-01-01T00:00:06+00:00,S,END\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gap"):
        energy.aggregate(
            csv_path,
            marker_path,
            max_boundary_gap_s=1.5,
        )


def test_energy_reaggregation_rejects_counter_regression(tmp_path):
    csv_path = tmp_path / "energy.csv"
    marker_path = tmp_path / "markers.csv"

    csv_path.write_text(
        "timestamp_utc,gpu0_energy_mJ,gpu1_energy_mJ\n"
        "2026-01-01T00:00:00+00:00,10,10\n"
        "2026-01-01T00:00:01+00:00,9,11\n",
        encoding="utf-8",
    )
    marker_path.write_text(
        "2026-01-01T00:00:00+00:00,S,START\n2026-01-01T00:00:01+00:00,S,END\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="counter decreased"):
        energy.aggregate(csv_path, marker_path)


def test_linear_exact_first_and_last_samples(tmp_path):
    energy_csv = tmp_path / "exact_energy.csv"
    markers_csv = tmp_path / "exact_markers.csv"

    energy_csv.write_text(
        "timestamp_utc,gpu0_energy_mJ,gpu1_energy_mJ\n"
        "2026-01-01T00:00:00Z,100,200\n"
        "2026-01-01T00:00:01Z,200,400\n"
        "2026-01-01T00:00:02Z,500,1000\n",
        encoding="utf-8",
    )

    markers_csv.write_text(
        "2026-01-01T00:00:00Z,EXACT,START\n2026-01-01T00:00:02Z,EXACT,END\n",
        encoding="utf-8",
    )

    result = energy.aggregate(
        energy_csv,
        markers_csv,
        boundary_policy="linear",
        max_boundary_gap_s=0.0,
    )

    stage = result["stages"][0]

    assert stage["gpu0_energy_mJ"] == 400
    assert stage["gpu1_energy_mJ"] == 800
    assert stage["gross_energy_mJ"] == 1200

    assert stage["counter_start_bracket_utc"] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    ]

    assert stage["counter_end_bracket_utc"] == [
        "2026-01-01T00:00:02Z",
        "2026-01-01T00:00:02Z",
    ]

    assert stage["start_boundary_max_bracketing_gap_seconds"] == 0
    assert stage["end_boundary_max_bracketing_gap_seconds"] == 0


def test_linear_exact_interior_sample(tmp_path):
    energy_csv = tmp_path / "interior_energy.csv"
    markers_csv = tmp_path / "interior_markers.csv"

    energy_csv.write_text(
        "timestamp_utc,gpu0_energy_mJ,gpu1_energy_mJ\n"
        "2026-01-01T00:00:00Z,100,200\n"
        "2026-01-01T00:00:01Z,200,400\n"
        "2026-01-01T00:00:02Z,500,1000\n",
        encoding="utf-8",
    )

    markers_csv.write_text(
        "2026-01-01T00:00:01Z,EXACT,START\n2026-01-01T00:00:02Z,EXACT,END\n",
        encoding="utf-8",
    )

    result = energy.aggregate(
        energy_csv,
        markers_csv,
        boundary_policy="linear",
        max_boundary_gap_s=0.0,
        idle_power_w=46.953,
    )

    stage = result["stages"][0]

    assert stage["gpu0_energy_mJ"] == 300
    assert stage["gpu1_energy_mJ"] == 600
    assert stage["marker_duration_seconds"] == 1.0

    assert stage["counter_start_bracket_utc"] == [
        "2026-01-01T00:00:01Z",
        "2026-01-01T00:00:01Z",
    ]

    assert stage["idle_energy_kWh"] == pytest.approx(46.953 / 3_600_000.0)


def test_linear_rejects_outside_observed_interval(tmp_path):
    energy_csv = tmp_path / "bounded_energy.csv"
    markers_csv = tmp_path / "bounded_markers.csv"

    energy_csv.write_text(
        "timestamp_utc,gpu0_energy_mJ,gpu1_energy_mJ\n"
        "2026-01-01T00:00:00Z,100,200\n"
        "2026-01-01T00:00:01Z,200,400\n",
        encoding="utf-8",
    )

    invalid_intervals = [
        (
            "2025-12-31T23:59:59Z",
            "2026-01-01T00:00:01Z",
        ),
        (
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:02Z",
        ),
    ]

    for start, end in invalid_intervals:
        markers_csv.write_text(
            f"{start},OUTSIDE,START\n{end},OUTSIDE,END\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="outside observed samples"):
            energy.aggregate(
                energy_csv,
                markers_csv,
                boundary_policy="linear",
            )


@pytest.mark.parametrize(
    "invalid_gap",
    [-1.0, float("nan"), float("inf")],
)
def test_rejects_invalid_boundary_gap(tmp_path, invalid_gap):
    with pytest.raises(
        ValueError,
        match="maximum boundary gap must be finite and non-negative",
    ):
        energy.aggregate(
            tmp_path / "unused_energy.csv",
            tmp_path / "unused_markers.csv",
            boundary_policy="linear",
            max_boundary_gap_s=invalid_gap,
        )
