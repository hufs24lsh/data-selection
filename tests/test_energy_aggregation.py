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
    assert stage["gross_gpu_kWh"] == pytest.approx(
        18000 / 3_600_000_000.0
    )
    assert stage["marker_duration_seconds"] == pytest.approx(5.6)
    assert stage["start_boundary_gap_seconds"] == pytest.approx(0.2)
    assert stage["end_boundary_gap_seconds"] == pytest.approx(0.2)

    expected_idle = 30.0 * 5.6 / 3_600_000.0
    assert stage["idle_energy_kWh"] == pytest.approx(expected_idle)
    assert stage["idle_adjusted_gpu_kWh"] == pytest.approx(
        stage["gross_gpu_kWh"] - expected_idle
    )


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
        "2026-01-01T00:00:04+00:00,S,START\n"
        "2026-01-01T00:00:06+00:00,S,END\n",
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
        "2026-01-01T00:00:00+00:00,S,START\n"
        "2026-01-01T00:00:01+00:00,S,END\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="counter decreased"):
        energy.aggregate(csv_path, marker_path)
