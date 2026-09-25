"""Step 1: GridGuard AI v2 - synthetic dataset generator.

Generates a completely fresh balanced dataset (no previous data or logic
reused) for a 5-class temperature-behavior classifier.

Design decisions:
  * Starting environmental temperature is random for every example.
  * Baseline window = 20 samples around that environment.
  * Current window = 20 samples generated from one of five behavior classes.
  * Generators use randomised slopes, rates, shapes, noise and amplitudes so
    classes overlap realistically and simple fixed-slope/threshold cheating is
    impossible.
  * Global seed configured in config.py -> reproducible output.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _generator_helpers import GENERATORS, generate_class_examples
from config import (
    BASE_COLUMNS,
    CLASS_NAMES,
    CURRENT_COLUMNS,
    DATASET_CSV,
    DATASET_META_JSON,
    EXAMPLES_PER_CLASS,
    RANDOM_SEED,
    START_TEMP_MAX,
    START_TEMP_MIN,
    WINDOW_LEN,
)

W = WINDOW_LEN


def build_dataset(seed=RANDOM_SEED, per_class=EXAMPLES_PER_CLASS):
    rng = np.random.default_rng(seed)
    x_list, y_list = [], []
    for class_id in range(len(GENERATORS)):
        x_class = generate_class_examples(rng, class_id, per_class)
        x_list.append(x_class)
        y_list.append(np.full(per_class, class_id, dtype=np.int64))
        print(f"  generated {per_class} examples for {GENERATORS[class_id][0]}")
    X = np.vstack(x_list)
    y = np.concatenate(y_list)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def quick_overlap_check(X, y):
    print("\nOverlap sanity check (max |current - baseline| distribution per class):")
    base = X[:, :W]
    curr = X[:, W:]
    delta = np.max(np.abs(curr - base), axis=1)
    print(f"  {'class':<24} {'count':>6} {'p10':>7} {'p50':>7} {'p90':>7}")
    for c in sorted(CLASS_NAMES):
        d = delta[y == c]
        p10, p50, p90 = np.percentile(d, [10, 50, 90])
        print(f"  {CLASS_NAMES[c]:<24} {int((y == c).sum()):>6} {p10:>7.2f} {p50:>7.2f} {p90:>7.2f}")
    print("\n  Overlapping design zones -> RAPID(p10) should be near DRIFT(p90):")
    print(f"  DRIFT  p90  = {np.percentile(delta[y == 2], 90):.2f} degC")
    print(f"  RAPID  p10  = {np.percentile(delta[y == 3], 10):.2f} degC")


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 1: Synthetic dataset generation (from zero)")
    print("=" * 72)
    print(f"Seed: {RANDOM_SEED} | Examples/class: {EXAMPLES_PER_CLASS} "
          f"| Window: {W} | Start temp range: [{START_TEMP_MIN}, {START_TEMP_MAX}] degC")

    X, y = build_dataset()
    df = pd.DataFrame(X, columns=BASE_COLUMNS + CURRENT_COLUMNS)
    df["label"] = y
    df.to_csv(DATASET_CSV, index=False)

    total = len(df)
    counts = df["label"].value_counts().sort_index()
    print(f"\nDataset written: {DATASET_CSV}")
    print(f"Total examples: {total}")
    print("Class balance:")
    for c, name in sorted(CLASS_NAMES.items()):
        print(f"  {c} = {name:<24} {int(counts[c]):>6}")

    meta = {
        "seed": RANDOM_SEED,
        "examples_per_class": EXAMPLES_PER_CLASS,
        "total_examples": int(total),
        "window_len": W,
        "features_per_row": 2 * W,
        "start_temp_range": [START_TEMP_MIN, START_TEMP_MAX],
        "class_counts": {int(c): int(n) for c, n in counts.items()},
        "class_names": CLASS_NAMES,
    }
    with open(DATASET_META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written: {DATASET_META_JSON}")

    quick_overlap_check(X, y)


if __name__ == "__main__":
    main()