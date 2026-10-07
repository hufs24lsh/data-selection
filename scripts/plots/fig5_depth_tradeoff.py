import matplotlib.pyplot as plt
from common import setup, save

setup()

data = [
    {"name": "K32",  "token_reduction": 66.82, "t1_overlap": 14.10},
    {"name": "K64",  "token_reduction": 59.42, "t1_overlap": 35.75},
    {"name": "K128", "token_reduction": 46.92, "t1_overlap": 62.55},
    {"name": "K256", "token_reduction": 29.14, "t1_overlap": 79.05},
]

fig, ax = plt.subplots(figsize=(7.8, 5.6))

xs = [d["token_reduction"] for d in data]
ys = [d["t1_overlap"] for d in data]

ax.plot(xs, ys, marker="o", linewidth=1.8)

for d in data:
    ax.annotate(
        d["name"],
        (d["token_reduction"], d["t1_overlap"]),
        xytext=(4, 6),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.8)
    )

ax.set_xlabel("Scoring Token Reduction (%)")
ax.set_ylabel("Raw-prefix T1 Selection Overlap (%)")
ax.set_title("Prefix Depth Trade-off")
ax.grid(True, alpha=0.3)

ax.set_xlim(25, 70)
ax.set_ylim(0, 90)

save(fig, "figs/prefix_depth_tradeoff.png")
print("Saved: figs/prefix_depth_tradeoff.png")
