import matplotlib.pyplot as plt
from common import setup, save

setup()

data = [
    {
        "name": "T1",
        "energy_reduction": 0.00,
        "macro": 47.1000,
        "token_reduction": 0.00,
    },
    {
        "name": "K256",
        "energy_reduction": 5.70,
        "macro": 46.6894,
        "token_reduction": 29.14,
    },
    {
        "name": "K128",
        "energy_reduction": 11.37,
        "macro": 46.6057,
        "token_reduction": 46.92,
    },
]

THRESHOLD = 46.60

fig, ax = plt.subplots(figsize=(8.2, 5.6))

# Utility-preserving region
ax.axhspan(
    THRESHOLD, 47.5,
    alpha=0.08,
    zorder=0,
)
ax.axhline(
    THRESHOLD,
    linestyle="--",
    linewidth=1.2,
)

# Direction of improvement:
# right = more energy saved
xs = [d["energy_reduction"] for d in data]
ys = [d["macro"] for d in data]

ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=8)

offsets = {
    "T1":   (12, 10),
    "K256": (12, 12),
    "K128": (-125, 14),
}

for d in data:
    dx, dy = offsets[d["name"]]

    if d["name"] == "T1":
        text = (
            f'T1 baseline\n'
            f'Macro {d["macro"]:.2f}'
        )
    else:
        text = (
            f'{d["name"]}\n'
            f'Macro {d["macro"]:.2f}\n'
            f'{d["energy_reduction"]:.2f}% less E2E energy\n'
            f'{d["token_reduction"]:.2f}% fewer scoring tokens'
        )

    ax.annotate(
        text,
        (d["energy_reduction"], d["macro"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="black",
            lw=0.8,
        ),
        arrowprops=dict(
            arrowstyle="-",
            lw=0.8,
        ),
    )

ax.text(
    11.8,
    THRESHOLD + 0.08,
    "Utility-preserving region",
    ha="right",
    va="bottom",
    fontsize=9,
)

ax.text(
    11.8,
    THRESHOLD - 0.08,
    "Threshold = 46.60",
    ha="right",
    va="top",
    fontsize=9,
)

ax.set_xlabel("End-to-End GPU Energy Reduction vs. T1 (%)")
ax.set_ylabel("Macro Score")
ax.set_title("Utility-Preserving Energy Reduction")

# Wider y-range so performance differences are not visually exaggerated
ax.set_xlim(-0.7, 12.5)
ax.set_ylim(44.0, 47.5)

ax.grid(True, alpha=0.22)

save(fig, "figs/performance_energy_pareto.png")
print("Saved: figs/performance_energy_pareto.png")
