import json
from pathlib import Path

import matplotlib.pyplot as plt
from common import METHOD_COLORS, save, setup

setup()

ROOT = Path(__file__).resolve().parents[2]
FACTOR_PATH = ROOT / "results" / "energy" / "carbon_factor_status.json"

factor_meta = json.loads(FACTOR_PATH.read_text(encoding="utf-8"))
EMISSION_FACTOR = float(factor_meta["kgCO2eq_per_kWh"])
FACTOR_STATUS = factor_meta["status"]

t1_energy = 0.682982
k256_energy = 0.644048
k128_energy = 0.605340

save_per_run_k256 = (t1_energy - k256_energy) * EMISSION_FACTOR
save_per_run_k128 = (t1_energy - k128_energy) * EMISSION_FACTOR

runs = [1, 10, 50, 100, 500, 1000]
k256_saved = [save_per_run_k256 * r for r in runs]
k128_saved = [save_per_run_k128 * r for r in runs]

fig, ax = plt.subplots(figsize=(7.8, 5.3))

ax.plot(
    runs,
    k256_saved,
    marker="o",
    linewidth=2.0,
    color=METHOD_COLORS["K256"],
    label="K256 vs T1",
)
ax.plot(
    runs,
    k128_saved,
    marker="o",
    linewidth=2.0,
    color=METHOD_COLORS["K128"],
    label="K128 vs T1",
)

ax.annotate(
    f"{k256_saved[-1]:.2f} kg",
    (runs[-1], k256_saved[-1]),
    xytext=(8, -10),
    textcoords="offset points",
    fontsize=9,
)
ax.annotate(
    f"{k128_saved[-1]:.2f} kg",
    (runs[-1], k128_saved[-1]),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=9,
)

ax.set_xlabel("Number of repeated runs")
ax.set_ylabel("Avoided CO$_2$eq (kg)")
ax.set_title(
    "Scenario: Cumulative Operational CO$_2$eq Reduction",
    fontweight="bold",
)
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

fig.text(
    0.5,
    0.01,
    (
        f"Scenario conversion: {EMISSION_FACTOR:.4f} kgCO2eq/kWh; "
        f"factor provenance status: {FACTOR_STATUS}. "
        "GPU-only operational-energy scope."
    ),
    ha="center",
    fontsize=8,
)

save(fig, "figs/co2_scaleout_scenario.png")
print("Saved: figs/co2_scaleout_scenario.png")
