import csv
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml

if len(sys.argv) != 2:
    raise SystemExit("usage: energy_logger.py OUTPUT.csv")

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)

if out.exists():
    raise RuntimeError(f"refusing to overwrite {out}")

running = True

def stop_handler(signum, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, stop_handler)
signal.signal(signal.SIGINT, stop_handler)

pynvml.nvmlInit()

try:
    n = pynvml.nvmlDeviceGetCount()
    if n != 2:
        raise RuntimeError(f"expected exactly 2 GPUs, found {n}")

    handles = [
        pynvml.nvmlDeviceGetHandleByIndex(i)
        for i in range(2)
    ]

    # Fail immediately if cumulative energy counters are unavailable.
    for h in handles:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(h)

    fields = [
        "timestamp_utc",
        "gpu0_energy_mJ",
        "gpu1_energy_mJ",
        "gpu0_power_mW",
        "gpu1_power_mW",
        "gpu0_util_pct",
        "gpu1_util_pct",
        "gpu0_mem_used_bytes",
        "gpu1_mem_used_bytes",
    ]

    with out.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()

        next_tick = time.monotonic()

        while running:
            ts = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )

            row = {"timestamp_utc": ts}

            for i, h in enumerate(handles):
                energy = int(
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                )
                power = int(
                    pynvml.nvmlDeviceGetPowerUsage(h)
                )
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)

                row[f"gpu{i}_energy_mJ"] = energy
                row[f"gpu{i}_power_mW"] = power
                row[f"gpu{i}_util_pct"] = int(util.gpu)
                row[f"gpu{i}_mem_used_bytes"] = int(mem.used)

            writer.writerow(row)
            f.flush()

            next_tick += 1.0
            delay = next_tick - time.monotonic()

            if delay > 0:
                time.sleep(delay)
            else:
                # Do not burst-sample after temporary scheduler delays.
                next_tick = time.monotonic()

finally:
    pynvml.nvmlShutdown()
