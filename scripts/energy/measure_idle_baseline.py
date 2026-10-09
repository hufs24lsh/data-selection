import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml

ROOT = Path.cwd()
OUT = ROOT / "outputs/energy"
OUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT / "device_energy_1hz.csv"
RESULT = OUT / "idle_baseline_300s.json"
MARKERS = OUT / "idle_markers.csv"
PROTOCOL = ROOT / "results/energy/archive/energy_measurement_protocol.json"

EXPECTED_PROTOCOL = (
    "b52f3a4a2dc83254ff63f0c16f6553934c7b3bf94949e2fa7bdd2a9571d9e350"
)

DURATION = 300


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.isoformat(timespec="milliseconds")


if RESULT.exists():
    raise RuntimeError(f"refusing to overwrite {RESULT}")

if sha256(PROTOCOL) != EXPECTED_PROTOCOL:
    raise RuntimeError("energy protocol SHA mismatch")

if not CSV_PATH.is_file():
    raise RuntimeError("energy logger CSV missing")


pynvml.nvmlInit()

try:
    handles = [
        pynvml.nvmlDeviceGetHandleByIndex(i)
        for i in range(2)
    ]

    def compute_pids():
        found = set()

        for h in handles:
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except Exception:
                procs = []

            for p in procs:
                found.add(int(p.pid))

        return sorted(found)

    before = compute_pids()

    if before:
        raise RuntimeError(
            f"GPU compute process present before idle baseline: {before}"
        )

    start = now()

    with MARKERS.open("a", encoding="utf-8") as f:
        f.write(f"{iso(start)},IDLE_BASELINE,START\n")
        f.flush()

    print("=" * 78, flush=True)
    print("IDLE BASELINE START", flush=True)
    print("duration_seconds =", DURATION, flush=True)
    print("start_utc =", iso(start), flush=True)
    print("=" * 78, flush=True)

    invalid_pids = set()

    for sec in range(DURATION):
        pids = compute_pids()

        if pids:
            invalid_pids.update(pids)
            break

        if (sec + 1) % 60 == 0:
            print(
                f"idle verified: {sec + 1}/{DURATION} s",
                flush=True,
            )

        time.sleep(1.0)

    end = now()

    valid = not invalid_pids and (end - start).total_seconds() >= 299

    with MARKERS.open("a", encoding="utf-8") as f:
        status = "END" if valid else "INVALID"
        f.write(
            f"{iso(end)},IDLE_BASELINE,{status}"
            + (
                ""
                if valid
                else f",compute_pids={sorted(invalid_pids)}"
            )
            + "\n"
        )
        f.flush()

finally:
    pynvml.nvmlShutdown()


if not valid:
    result = {
        "status": "INVALID",
        "reason": "GPU compute process detected during idle window",
        "compute_pids": sorted(invalid_pids),
        "start_utc": iso(start),
        "end_utc": iso(end),
    }

    RESULT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    raise SystemExit(2)


# Give logger one extra sample after END boundary.
time.sleep(2.0)

rows = []

with CSV_PATH.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ts = datetime.fromisoformat(row["timestamp_utc"])

        if start <= ts <= end:
            rows.append(row)

if len(rows) < 295:
    raise RuntimeError(
        f"too few logger samples inside idle window: {len(rows)}"
    )

first = rows[0]
last = rows[-1]

e0_mj = (
    int(last["gpu0_energy_mJ"])
    - int(first["gpu0_energy_mJ"])
)

e1_mj = (
    int(last["gpu1_energy_mJ"])
    - int(first["gpu1_energy_mJ"])
)

gross_mj = e0_mj + e1_mj

sample_duration_s = (
    datetime.fromisoformat(last["timestamp_utc"])
    - datetime.fromisoformat(first["timestamp_utc"])
).total_seconds()

gross_kwh = gross_mj / 3_600_000_000.0
counter_mean_power_w = (
    gross_mj / 1000.0 / sample_duration_s
)

mean_sample_power_w = sum(
    (
        int(r["gpu0_power_mW"])
        + int(r["gpu1_power_mW"])
    )
    / 1000.0
    for r in rows
) / len(rows)

mean_gpu0_power_w = sum(
    int(r["gpu0_power_mW"]) / 1000.0
    for r in rows
) / len(rows)

mean_gpu1_power_w = sum(
    int(r["gpu1_power_mW"]) / 1000.0
    for r in rows
) / len(rows)

result = {
    "status": "PASS",
    "scope": "GPU_ONLY",
    "duration_requested_seconds": DURATION,
    "start_utc": iso(start),
    "end_utc": iso(end),
    "logger_samples": len(rows),
    "sample_duration_seconds": sample_duration_s,
    "gpu0_energy_mJ": e0_mj,
    "gpu1_energy_mJ": e1_mj,
    "gross_energy_mJ": gross_mj,
    "gross_energy_kWh": gross_kwh,
    "counter_implied_mean_two_gpu_power_W":
        counter_mean_power_w,
    "sampled_mean_gpu0_power_W":
        mean_gpu0_power_w,
    "sampled_mean_gpu1_power_W":
        mean_gpu1_power_w,
    "sampled_mean_two_gpu_power_W":
        mean_sample_power_w,
    "primary_idle_power_W":
        counter_mean_power_w,
    "primary_idle_power_source":
        "NVML cumulative energy counter difference",
    "protocol_sha256": EXPECTED_PROTOCOL,
}

RESULT.write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 78)
print("IDLE BASELINE RESULT")
print("=" * 78)
print(json.dumps(result, indent=2))
print("idle_result_sha256 =", sha256(RESULT))
