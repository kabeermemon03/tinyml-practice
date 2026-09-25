"""Step 3: GridGuard AI v2 - preprocessing.

Stratified 70/15/15 split with NO leakage:
  * scaler fitted ONLY on the training split
  * validation and test windows transformed with the same train-fitted scaler
  * normalization is a per-feature global z-score over the 40 inputs, which
    keeps absolute temperature information (it does not remove magnitude)
  * parameters (mean, scale) saved for the ESP32 to reproduce later
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLASS_NAMES,
    DATASET_CSV,
    FEATURE_COLUMNS,
    NORM_PARAMS_JSON,
    NORM_PARAMS_NPZ,
    RANDOM_SEED,
    TEST_NPZ,
    TEST_RATIO,
    TRAIN_NPZ,
    TRAIN_RATIO,
    VAL_NPZ,
    VAL_RATIO,
)


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 3: Data preprocessing (split + scaling)")
    print("=" * 72)
    df = pd.read_csv(DATASET_CSV)
    X = df[FEATURE_COLUMNS].to_numpy().astype(np.float32)
    y = df["label"].to_numpy().astype(np.int64)
    print(f"Loaded {len(df)} examples, {X.shape[1]} features")

    train_ratio = TRAIN_RATIO + VAL_RATIO
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=VAL_RATIO / train_ratio,
        random_state=RANDOM_SEED, stratify=y_tr,
    )
    print(f"Splits -> train {len(X_train)} | val {len(X_val)} | test {len(X_te)}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_te).astype(np.float32)

    np.savez_compressed(TRAIN_NPZ, X=X_train_s, y=y_train)
    np.savez_compressed(VAL_NPZ, X=X_val_s, y=y_val)
    np.savez_compressed(TEST_NPZ, X=X_test_s, y=y_te)
    np.savez_compressed(NORM_PARAMS_NPZ, mean=mean, scale=scale)

    params = {
        "method": "per_feature_zscore",
        "formula": "x_norm[i] = (x_raw[i] - mean[i]) / scale[i] for i in 0..39",
        "fitted_on": "training split only",
        "n_features": int(len(mean)),
        "mean": [round(float(v), 6) for v in mean],
        "scale": [round(float(v), 6) for v in scale],
    }
    with open(NORM_PARAMS_JSON, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nSaved train/val/test: {TRAIN_NPZ}, {VAL_NPZ}, {TEST_NPZ}")
    print(f"Saved normalization parameters: {NORM_PARAMS_NPZ} + {NORM_PARAMS_JSON}")
    print("Normalization method: per-feature z-score (global, preserves absolute temperature)")
    print("mean[:4]  =", np.round(mean[:4], 4))
    print("scale[:4] =", np.round(scale[:4], 4))
    print("mean[-4:]  =", np.round(mean[-4:], 4))
    print("scale[-4:] =", np.round(scale[-4:], 4))

    print("\nClass distribution check (must be balanced in every split):")
    for name, yy in (("train", y_train), ("val", y_val), ("test", y_te)):
        counts = np.bincount(yy, minlength=len(CLASS_NAMES))
        print(f"  {name:<6} " + " ".join(f"{CLASS_NAMES[c][:8]}={n}" for c, n in enumerate(counts)))


if __name__ == "__main__":
    main()