"""GridGuard Fire AI - Step 4: Dataset visualization.

Produces professional figures describing the raw dataset and the split:
  results/figures/dataset_class_distribution.png
  results/figures/dataset_samples.png
  results/figures/dataset_dimensions.png
  results/figures/dataset_color_stats.png
  results/figures/dataset_duplicates.png
  results/figures/dataset_split_distribution.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CLEAN_MANIFEST,
    CLASS_NAMES,
    FIGURES_DIR,
    RANDOM_SEED,
    SPLITS_MANIFEST,
)
from _helpers import load_manifest  # noqa: E402

import random


def _sample_paths(rows, label, k, rng):
    pool = [r["path"] for r in rows if int(r["label"]) == label]
    return rng.sample(pool, min(k, len(pool)))


def _plot_samples(rows):
    rng = random.Random(RANDOM_SEED)
    k = 8
    fig, axes = plt.subplots(2, k, figsize=(2 * k + 2, 6.5))
    for row_idx, label in enumerate([0, 1]):
        paths = _sample_paths(rows, label, k, rng)
        for col, p in enumerate(paths):
            ax = axes[row_idx, col]
            img = Image.open(p)
            img = ImageOps.exif_transpose(img).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("NO_FIRE", fontsize=12)
    axes[1, 0].set_ylabel("FIRE", fontsize=12)
    fig.suptitle("Raw dataset samples (labels as shipped)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_samples.png", dpi=110)
    plt.close(fig)
    print("  dataset_samples.png")


def _plot_class_distribution(rows):
    counts = {0: sum(1 for r in rows if int(r["label"]) == 0),
              1: sum(1 for r in rows if int(r["label"]) == 1)}
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["NO_FIRE", "FIRE"]
    ax.bar(labels, [counts[0], counts[1]], color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("image count")
    ax.set_title("Class distribution (raw dataset)")
    for i, v in enumerate([counts[0], counts[1]]):
        ax.text(i, v + 80, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_class_distribution.png", dpi=110)
    plt.close(fig)
    print("  dataset_class_distribution.png")


def _plot_dimensions(rows):
    widths = [int(r["width"]) for r in rows]
    heights = [int(r["height"]) for r in rows]
    aspects = [float(r["aspect_ratio"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(widths, bins=40, color="#4C72B0")
    axes[0].set_title("Width distribution"); axes[0].set_xlabel("pixels")
    axes[1].hist(heights, bins=40, color="#55A868")
    axes[1].set_title("Height distribution"); axes[1].set_xlabel("pixels")
    axes[2].hist(aspects, bins=40, color="#C44E52")
    axes[2].set_title("Aspect ratio (w/h)"); axes[2].set_xlabel("ratio")
    fig.suptitle("Raw dimension analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_dimensions.png", dpi=110)
    plt.close(fig)
    print("  dataset_dimensions.png")


def _plot_color_stats(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        b = [float(r["brightness"]) for r in rows if int(r["label"]) == label]
        axes[0].hist(b, bins=40, alpha=0.6, label=CLASS_NAMES[label], color=color)
        rd = [float(r["redness"]) for r in rows if int(r["label"]) == label]
        axes[1].hist(rd, bins=40, alpha=0.6, label=CLASS_NAMES[label], color=color)
    axes[0].set_title("Mean brightness"); axes[0].legend()
    axes[1].set_title("Mean redness (R - B)"); axes[1].legend()
    fig.suptitle("Color statistics by class - helps detect 'orange=fire' shortcut risk",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_color_stats.png", dpi=110)
    plt.close(fig)
    print("  dataset_color_stats.png")


def _plot_duplicates():
    import json
    from config import DUPLICATE_GROUPS_JSON, NEAR_DUPLICATE_CSV, REPORTS_DIR
    import csv
    with open(DUPLICATE_GROUPS_JSON, encoding="utf-8") as fh:
        dup = json.load(fh)
    sizes = [len(v) for v in dup.values() if len(v) > 1]
    n_near = 0
    near_path = REPORTS_DIR / "near_duplicate_pairs.csv"
    if near_path.exists():
        with open(near_path, newline="", encoding="utf-8") as fh:
            n_near = sum(1 for _ in csv.DictReader(fh))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["Exact\ndup groups", "Near-dup\npairs"], [len(sizes), n_near],
                color=["#C44E52", "#55A868"])
    axes[0].set_title("Duplicate detection")
    axes[1].hist(sizes, bins=max(2, max(sizes, default=1)), color="#C44E52")
    axes[1].set_title("Images per duplicate group")
    axes[1].set_xlabel("group size")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_duplicates.png", dpi=110)
    plt.close(fig)
    print("  dataset_duplicates.png")


def _plot_split_distribution():
    import csv
    from config import SPLITS_MANIFEST
    if not SPLITS_MANIFEST.exists():
        print("  (split not found - run 03 first)")
        return
    counts = {}
    with open(SPLITS_MANIFEST, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            counts.setdefault(r["split"], {0: 0, 1: 0})
            counts[r["split"]][int(r["label"])] += 1
    splits = ["train", "validation", "test"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    xs = range(len(splits))
    ax.bar([x - width / 2 for x in xs], [counts[s][0] for s in splits], width,
           label="NO_FIRE", color="#4C72B0")
    ax.bar([x + width / 2 for x in xs], [counts[s][1] for s in splits], width,
           label="FIRE", color="#DD8452")
    ax.set_xticks(list(xs)); ax.set_xticklabels(splits)
    ax.set_ylabel("image count"); ax.legend()
    ax.set_title("Split distribution (by manifest, leakage-safe)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_split_distribution.png", dpi=110)
    plt.close(fig)
    print("  dataset_split_distribution.png")


def main():
    print("=== GridGuard Fire AI | Step 4: visualization ===")
    rows = load_manifest(CLEAN_MANIFEST)
    print(f"Cleaned images available: {len(rows)}")
    _plot_samples(rows)
    _plot_class_distribution(rows)
    _plot_dimensions(rows)
    _plot_color_stats(rows)
    _plot_duplicates()
    _plot_split_distribution()
    print(f"\nFigures saved to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())