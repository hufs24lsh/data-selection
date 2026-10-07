import matplotlib.pyplot as plt
from common import setup, save

setup()

stages = ["Calibration", "Scoring", "Final FT"]

series = {
    "T1":        [0.189114, 0.228271, 0.265597],
    "K256":      [0.189114, 0.174823, 0.280111],
    "true-K128": [0.189114, 0.141707, 0.274520],
}

fig, ax = plt.subplots(figsize=(8.0, 5.4))
x = list(range(len(stages)))

for name, vals in series.items():
    ax.plot(x, vals, marker="o", linewidth=1.8, label=name)
    for xi, yi in zip(x, vals):
        ax.annotate(
            f"{yi:.3f}",
            (xi, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylabel("Stage Energy (kWh)")
ax.set_xlabel("Pipeline Stage")
ax.set_title("Stage-wise Energy Consumption")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=True)

ax.set_ylim(0.12, 0.31)

save(fig, "figs/stagewise_energy.png")
print("Saved: figs/stagewise_energy.png")
