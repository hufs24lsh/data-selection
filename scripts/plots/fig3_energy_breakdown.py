import numpy as np
import matplotlib.pyplot as plt

from common import METHOD_COLORS, setup, save

setup()

stages = ["Calibration", "Scoring", "Final FT"]
x = np.arange(len(stages), dtype=float)

# Historical, validated stage energies (gross GPU kWh)
series = {
    "T1": [0.1891142217063904, 0.22827087977040608, 0.26559736145665486],
    "K256": [0.1891142217063904, 0.17482281972078959, 0.2801107961055501],
    "K128": [0.1891142217063904, 0.1417066632012431, 0.2745195120864953],
}

totals = {
    "T1": 0.6829824629521264,
    "K256": 0.6440478375413852,
    "K128": 0.6053403969941372,
}

t1 = np.array(series["T1"], dtype=float)
k256 = np.array(series["K256"], dtype=float)
k128 = np.array(series["K128"], dtype=float)

fig, ax = plt.subplots(figsize=(8.0, 5.2))

# --- filled regions ---
# Under K128
ax.fill_between(
    x,
    0,
    k128,
    color=METHOD_COLORS["K128"],
    alpha=0.16,
    zorder=1,
)

# Between K128 and K256
ax.fill_between(
    x,
    np.minimum(k128, k256),
    np.maximum(k128, k256),
    color=METHOD_COLORS["K256"],
    alpha=0.14,
    zorder=1,
)

# Between K256 and T1
ax.fill_between(
    x,
    np.minimum(k256, t1),
    np.maximum(k256, t1),
    color=METHOD_COLORS["T1"],
    alpha=0.12,
    zorder=1,
)

# --- lines ---
for name in ["K128", "K256", "T1"]:
    y = np.array(series[name], dtype=float)
    ax.plot(
        x,
        y,
        marker="o",
        markersize=6.5,
        linewidth=2.4,
        color=METHOD_COLORS[name],
        label=name,
        zorder=3,
    )

# --- total annotations near the last point only ---
label_offsets = {
    "K128": (10, -4),
    "K256": (10, 10),
    "T1": (10, -18),
}

for name in ["K128", "K256", "T1"]:
    y = np.array(series[name], dtype=float)
    dx, dy = label_offsets[name]
    ax.annotate(
        f"Total {totals[name]:.3f} kWh",
        (x[-1], y[-1]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color=METHOD_COLORS[name],
        fontweight="semibold",
    )

ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylabel("Stage Energy (kWh)")
ax.set_xlabel("Pipeline Stage")
ax.set_title("Stage-wise Energy Consumption", fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)

ax.set_ylim(0.00, 0.31)
ax.set_xlim(-0.08, 2.35)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper left",
)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
save(fig, "figs/stagewise_energy.png")
print("Saved: figs/stagewise_energy.png")
