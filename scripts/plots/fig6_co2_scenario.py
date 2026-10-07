import matplotlib.pyplot as plt
from common import setup, save

setup()

EMISSION_FACTOR = 0.4541  # kgCO2eq / kWh

t1_energy = 0.682982
k256_energy = 0.644048
k128_energy = 0.605340

save_per_run_k256 = (t1_energy - k256_energy) * EMISSION_FACTOR
save_per_run_k128 = (t1_energy - k128_energy) * EMISSION_FACTOR

runs = [1, 10, 50, 100, 500, 1000]
k256_saved = [save_per_run_k256 * r for r in runs]
k128_saved = [save_per_run_k128 * r for r in runs]

fig, ax = plt.subplots(figsize=(7.8, 5.3))

ax.plot(runs, k256_saved, marker="o", linewidth=1.8, label="K256 vs T1")
ax.plot(runs, k128_saved, marker="o", linewidth=1.8, label="true-K128 vs T1")

# 끝 점만 라벨
ax.annotate(
    f"{k256_saved[-1]:.2f} kg",
    (runs[-1], k256_saved[-1]),
    xytext=(8, -10),
    textcoords="offset points",
    fontsize=9
)
ax.annotate(
    f"{k128_saved[-1]:.2f} kg",
    (runs[-1], k128_saved[-1]),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=9
)

ax.set_xlabel("Number of repeated runs")
ax.set_ylabel("Avoided CO$_2$eq (kg)")
ax.set_title("Estimated Cumulative CO$_2$ Reduction")
ax.grid(True, alpha=0.25)
ax.legend(frameon=True)

save(fig, "figs/co2_scaleout_scenario.png")
print("Saved: figs/co2_scaleout_scenario.png")
