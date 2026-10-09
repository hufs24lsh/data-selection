#!/usr/bin/env python3
"""Aggregate GPU energy from NVML counter logs and stage markers.

Input logs are read without modifying reported experimental results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from datetime import datetime
from pathlib import Path


def parse_time(value: str) -> datetime:
    # Python 3.10 requires an explicit UTC offset instead of Z.
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)


def load_energy_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        raise ValueError("energy log must contain at least two samples")

    required = {
        "timestamp_utc",
        "gpu0_energy_mJ",
        "gpu1_energy_mJ",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"energy log missing columns: {sorted(missing)}")

    parsed = []
    previous_time = None
    previous_e0 = None
    previous_e1 = None

    for lineno, row in enumerate(rows, start=2):
        try:
            ts = parse_time(row["timestamp_utc"])
            e0 = int(row["gpu0_energy_mJ"])
            e1 = int(row["gpu1_energy_mJ"])
        except Exception as exc:
            raise ValueError(f"invalid energy row at CSV line {lineno}") from exc

        if previous_time is not None and ts <= previous_time:
            raise ValueError("energy timestamps must be strictly increasing")
        if previous_e0 is not None and e0 < previous_e0:
            raise ValueError("GPU0 cumulative energy counter decreased")
        if previous_e1 is not None and e1 < previous_e1:
            raise ValueError("GPU1 cumulative energy counter decreased")

        parsed.append(
            {
                "timestamp": ts,
                "timestamp_utc": row["timestamp_utc"],
                "gpu0_energy_mJ": e0,
                "gpu1_energy_mJ": e1,
            }
        )
        previous_time = ts
        previous_e0 = e0
        previous_e1 = e1

    return parsed


def load_markers(path: Path) -> dict[str, dict[str, datetime]]:
    markers: dict[str, dict[str, datetime]] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
            if not row:
                continue
            if row[0].strip() == "timestamp_utc":
                continue
            if len(row) < 3:
                raise ValueError(
                    f"marker line {lineno} must have timestamp,stage,status"
                )

            ts = parse_time(row[0].strip())
            stage = row[1].strip()
            status = row[2].strip().upper()

            if status not in {"START", "END"}:
                # INVALID or diagnostic rows are not usable stage boundaries.
                continue

            stage_markers = markers.setdefault(stage, {})
            if status in stage_markers:
                raise ValueError(f"duplicate {status} marker for stage {stage}")
            stage_markers[status] = ts

    complete = {}
    for stage, values in markers.items():
        if set(values) != {"START", "END"}:
            raise ValueError(f"incomplete markers for stage {stage}")
        if values["END"] <= values["START"]:
            raise ValueError(f"non-positive marker duration for stage {stage}")
        complete[stage] = values

    if not complete:
        raise ValueError("no complete START/END stage markers found")

    return complete


def nearest_sample(rows: list[dict], target: datetime) -> tuple[dict, float]:
    times = [row["timestamp"] for row in rows]
    pos = bisect_left(times, target)

    candidates = []
    if pos < len(rows):
        candidates.append(rows[pos])
    if pos > 0:
        candidates.append(rows[pos - 1])

    sample = min(
        candidates,
        key=lambda row: abs((row["timestamp"] - target).total_seconds()),
    )
    gap = abs((sample["timestamp"] - target).total_seconds())
    return sample, gap


def summarize_stage(
    rows: list[dict],
    *,
    stage: str,
    start: datetime,
    end: datetime,
    max_boundary_gap_s: float,
    idle_power_w: float | None,
) -> dict:
    first, start_gap = nearest_sample(rows, start)
    last, end_gap = nearest_sample(rows, end)

    if start_gap > max_boundary_gap_s:
        raise ValueError(
            f"{stage}: start marker/sample gap {start_gap:.3f}s exceeds "
            f"{max_boundary_gap_s:.3f}s"
        )
    if end_gap > max_boundary_gap_s:
        raise ValueError(
            f"{stage}: end marker/sample gap {end_gap:.3f}s exceeds "
            f"{max_boundary_gap_s:.3f}s"
        )
    if last["timestamp"] <= first["timestamp"]:
        raise ValueError(f"{stage}: selected samples have non-positive duration")

    e0 = last["gpu0_energy_mJ"] - first["gpu0_energy_mJ"]
    e1 = last["gpu1_energy_mJ"] - first["gpu1_energy_mJ"]
    if e0 < 0 or e1 < 0:
        raise ValueError(f"{stage}: cumulative energy delta is negative")

    sample_duration = (last["timestamp"] - first["timestamp"]).total_seconds()
    marker_duration = (end - start).total_seconds()
    gross_mj = e0 + e1
    gross_kwh = gross_mj / 3_600_000_000.0

    result = {
        "stage": stage,
        "marker_start_utc": start.isoformat(),
        "marker_end_utc": end.isoformat(),
        "marker_duration_seconds": marker_duration,
        "counter_start_sample_utc": first["timestamp_utc"],
        "counter_end_sample_utc": last["timestamp_utc"],
        "start_boundary_gap_seconds": start_gap,
        "end_boundary_gap_seconds": end_gap,
        "counter_sample_duration_seconds": sample_duration,
        "gpu0_energy_mJ": e0,
        "gpu1_energy_mJ": e1,
        "gross_energy_mJ": gross_mj,
        "gross_gpu_kWh": gross_kwh,
    }

    if idle_power_w is not None:
        if not math.isfinite(idle_power_w) or idle_power_w < 0:
            raise ValueError("idle power must be finite and non-negative")
        idle_kwh = idle_power_w * marker_duration / 3_600_000.0
        result["idle_power_W"] = idle_power_w
        result["idle_energy_kWh"] = idle_kwh
        result["idle_adjusted_gpu_kWh"] = gross_kwh - idle_kwh

    return result


def interpolated_boundary(
    rows: list[dict],
    target: datetime,
    *,
    max_boundary_gap_s: float,
) -> dict:
    """Interpolate cumulative counters between observed 1 Hz samples."""
    times = [row["timestamp"] for row in rows]
    pos = bisect_left(times, target)

    # An observed counter at the exact marker time needs no interpolation.
    # This also handles the first and last samples without extrapolation.
    if pos < len(rows) and times[pos] == target:
        sample = rows[pos]
        return {
            "before_sample_utc": sample["timestamp_utc"],
            "after_sample_utc": sample["timestamp_utc"],
            "max_bracketing_gap_seconds": 0.0,
            "gpu0_energy_mJ": sample["gpu0_energy_mJ"],
            "gpu1_energy_mJ": sample["gpu1_energy_mJ"],
        }

    if pos <= 0 or pos >= len(rows):
        raise ValueError("interpolation boundary outside observed samples")

    before = rows[pos - 1]
    after = rows[pos]

    left_gap = (target - before["timestamp"]).total_seconds()
    right_gap = (after["timestamp"] - target).total_seconds()
    interval = (after["timestamp"] - before["timestamp"]).total_seconds()

    if interval <= 0 or left_gap < 0 or right_gap < 0:
        raise ValueError("invalid interpolation interval")

    if max(left_gap, right_gap) > max_boundary_gap_s:
        raise ValueError(
            f"interpolation boundary gap exceeds {max_boundary_gap_s:.3f}s"
        )

    fraction = left_gap / interval

    result = {
        "before_sample_utc": before["timestamp_utc"],
        "after_sample_utc": after["timestamp_utc"],
        "max_bracketing_gap_seconds": max(left_gap, right_gap),
    }

    for field in ("gpu0_energy_mJ", "gpu1_energy_mJ"):
        result[field] = before[field] + fraction * (after[field] - before[field])

    return result


def summarize_stage_linear(
    rows: list[dict],
    *,
    stage: str,
    start: datetime,
    end: datetime,
    max_boundary_gap_s: float,
    idle_power_w: float | None,
) -> dict:
    """Summarize a stage using interpolated boundary counter values."""
    first = interpolated_boundary(rows, start, max_boundary_gap_s=max_boundary_gap_s)
    last = interpolated_boundary(rows, end, max_boundary_gap_s=max_boundary_gap_s)

    e0 = last["gpu0_energy_mJ"] - first["gpu0_energy_mJ"]
    e1 = last["gpu1_energy_mJ"] - first["gpu1_energy_mJ"]

    if e0 < 0 or e1 < 0:
        raise ValueError(f"{stage}: interpolated counter decreased")

    marker_duration = (end - start).total_seconds()
    gross_mj = e0 + e1
    gross_kwh = gross_mj / 3_600_000_000.0

    result = {
        "stage": stage,
        "boundary_policy": "linear",
        "marker_start_utc": start.isoformat(),
        "marker_end_utc": end.isoformat(),
        "marker_duration_seconds": marker_duration,
        "counter_start_bracket_utc": [
            first["before_sample_utc"],
            first["after_sample_utc"],
        ],
        "counter_end_bracket_utc": [
            last["before_sample_utc"],
            last["after_sample_utc"],
        ],
        "start_boundary_max_bracketing_gap_seconds": first[
            "max_bracketing_gap_seconds"
        ],
        "end_boundary_max_bracketing_gap_seconds": last["max_bracketing_gap_seconds"],
        "gpu0_energy_mJ": e0,
        "gpu1_energy_mJ": e1,
        "gross_energy_mJ": gross_mj,
        "gross_gpu_kWh": gross_kwh,
    }

    if idle_power_w is not None:
        if not math.isfinite(idle_power_w) or idle_power_w < 0:
            raise ValueError("idle power must be finite and non-negative")
        idle_kwh = idle_power_w * marker_duration / 3_600_000.0
        result["idle_power_W"] = idle_power_w
        result["idle_energy_kWh"] = idle_kwh
        result["idle_adjusted_gpu_kWh"] = gross_kwh - idle_kwh

    return result


def aggregate(
    energy_csv: Path,
    markers_csv: Path,
    *,
    max_boundary_gap_s: float = 1.5,
    idle_power_w: float | None = None,
    boundary_policy: str = "nearest",
) -> dict:
    if boundary_policy not in {"nearest", "linear"}:
        raise ValueError("unsupported boundary policy")
    if not math.isfinite(max_boundary_gap_s) or max_boundary_gap_s < 0:
        raise ValueError("maximum boundary gap must be finite and non-negative")
    rows = load_energy_rows(energy_csv)
    markers = load_markers(markers_csv)

    stage_summarizer = (
        summarize_stage if boundary_policy == "nearest" else summarize_stage_linear
    )
    stages = [
        stage_summarizer(
            rows,
            stage=stage,
            start=values["START"],
            end=values["END"],
            max_boundary_gap_s=max_boundary_gap_s,
            idle_power_w=idle_power_w,
        )
        for stage, values in sorted(markers.items())
    ]

    return {
        "scope": "GPU_ONLY",
        "counter_source": "NVML cumulative energy mJ",
        "boundary_policy": boundary_policy,
        "max_boundary_gap_seconds": max_boundary_gap_s,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-csv", type=Path, required=True)
    parser.add_argument("--markers-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--idle-power-w", type=float)
    parser.add_argument("--max-boundary-gap-s", type=float, default=1.5)
    parser.add_argument(
        "--boundary-policy",
        choices=("nearest", "linear"),
        default="nearest",
    )
    args = parser.parse_args()

    result = aggregate(
        args.energy_csv,
        args.markers_csv,
        max_boundary_gap_s=args.max_boundary_gap_s,
        idle_power_w=args.idle_power_w,
        boundary_policy=args.boundary_policy,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"

    if args.output is not None:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite {args.output}")
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
