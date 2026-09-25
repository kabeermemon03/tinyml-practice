"""Step 2: GridGuard AI v2 - dataset visualization.

Shows example sequences from every class (baseline + current window) and the
overlap in behaviour so the user can verify the synthetic data looks real
before any model is trained.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLASS_NAMES,
    CURRENT_COLUMNS,
    DATASET_CSV,
    DATASET_OVERVIEW_PNG,
    FEATURE_COLUMNS,
    WINDOW_LEN,
)

W = WINDOW_LEN
N_EXAMPLES = 3
ACCENTS = ["#2e7d32", "#1565c0", "#8e24aa", "#c62828", "#ef6c00"]


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 2: Dataset visualization")
    print("=" * 72)
    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded {len(df)} examples from {DATASET_CSV}")

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label"].to_numpy()

    fig, axes = plt.subplots(len(CLASS_NAMES), N_EXAMPLES, figsize=(14, 12), sharex=True)
    for row, class_id in enumerate(sorted(CLASS_NAMES)):
        idx = np.where(y == class_id)[0][:N_EXAMPLES]
        for col, i in enumerate(idx):
            ax = axes[row, col]
            base = X[i, :W]
            curr = X[i, W:]
            t_base = np.arange(W)
            t_curr = np.arange(W - 1, 2 * W - 1)
            ax.plot(t_base, base, "o-", color=ACCENTS[class_id], label="baseline", markersize=3)
            ax.plot(t_curr, curr, "s-", color="k", label="current", markersize=3, linewidth=1.2)
            ax.axvline(W - 1, color="grey", linestyle="--", linewidth=0.8)
            ax.set_title(f"start={base[0]:.1f} C", fontsize=8)
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[class_id], fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Example sequences per class (first 20 = baseline | last 20 = current)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(DATASET_OVERVIEW_PNG, dpi=130)
    plt.close(fig)
    print(f"Example sequences saved: {DATASET_OVERVIEW_PNG}")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    delta = np.max(np.abs(X[:, W:] - X[:, :W]), axis=1)
    bins = np.linspace(0, 18, 60)
    for class_id, color in zip(sorted(CLASS_NAMES), ACCENTS):
        ax2.hist(delta[y == class_id], bins=bins, alpha=0.55, label=CLASS_NAMES[class_id], color=color)
    ax2.set_xlabel("max |current - baseline| over the window (degC)")
    ax2.set_ylabel("examples")
    ax2.set_title("Behaviour overlap: max displacement from baseline per class")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    overlap_png = DATASET_OVERVIEW_PNG.with_name("dataset_overlap_histogram.png")
    fig2.savefig(overlap_png, dpi=130)
    plt.close(fig2)
    print(f"Overlap histogram saved: {overlap_png}")


if __name__ == "__main__":
    main()