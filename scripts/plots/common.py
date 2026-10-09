from pathlib import Path

import matplotlib.pyplot as plt

METHOD_COLORS = {
    "K128": "#2E9B57",  # green
    "K256": "#4C78D0",  # blue
    "T1": "#D65F5F",  # red
}

DISPLAY_NAME = {
    "K128": "K128",
    "K256": "K256",
    "T1": "T1",
}


def setup():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        }
    )


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
