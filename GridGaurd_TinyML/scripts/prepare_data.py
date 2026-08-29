import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI
# MODEL 1 - DATA PREPARATION
# Temperature Anomaly Detection
# ============================================================

DATA_FILE = config.DATASET_FILE

RANDOM_SEED = config.RANDOM_SEED
WINDOW_SIZE = config.WINDOW_SIZE

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# LOAD DATASET
# ============================================================
print()
print("==========================================")
print(" GRIDGUARD AI - DATA PREPARATION")
print("==========================================")

print()
print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")

# ============================================================
# CHECK DATASET
# ============================================================
print()
print("Checking dataset...")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Normal samples : {(df['label'] == 0).sum()}")
print(f"Anomaly samples: {(df['label'] == 1).sum()}")

# ============================================================
# SEPARATE INPUTS AND LABELS
# ============================================================
temperature_columns = [f"temp_{i}" for i in range(1, WINDOW_SIZE + 1)]

X = df[temperature_columns].values
y = df["label"].values

print()
print("Input shape:", X.shape)
print("Label shape:", y.shape)

# ============================================================
# TRAIN / TEMPORARY SPLIT
#
# 70% Training
# 30% Temporary (Validation + Test)
# ============================================================
print()
print("Creating train/validation/test split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    random_state=RANDOM_SEED,
    stratify=y,
)

# ============================================================
# VALIDATION / TEST SPLIT
#
# Temporary 30%:
#   15% Validation
#   15% Test
# ============================================================
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp,
)

# ============================================================
# DISPLAY SPLIT
# ============================================================
print()
print("Dataset split:")
print(f"Training   : {len(X_train)} samples")
print(f"Validation : {len(X_val)} samples")
print(f"Testing    : {len(X_test)} samples")
print()
print(f"Training percentage   : {len(X_train) / len(X) * 100:.1f}%")
print(f"Validation percentage : {len(X_val) / len(X) * 100:.1f}%")
print(f"Testing percentage    : {len(X_test) / len(X) * 100:.1f}%")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================
print()
print("Class distribution:")
print()
print("Training:")
print(f" Normal  : {(y_train == 0).sum()}")
print(f" Anomaly : {(y_train == 1).sum()}")
print()
print("Validation:")
print(f" Normal  : {(y_val == 0).sum()}")
print(f" Anomaly : {(y_val == 1).sum()}")
print()
print("Testing:")
print(f" Normal  : {(y_test == 0).sum()}")
print(f" Anomaly : {(y_test == 1).sum()}")

# ============================================================
# NORMALIZATION
#
# IMPORTANT:
# Fit ONLY on training data to prevent information leakage
# from the validation and test datasets.
# ============================================================
print()
print("Normalizing data...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ============================================================
# SAVE SCALER PARAMETERS
# ============================================================
print()
print("Saving scaler parameters...")

np.save(config.SCALER_MEAN_FILE, scaler.mean_)
np.save(config.SCALER_SCALE_FILE, scaler.scale_)

# ============================================================
# FINAL SHAPES
# ============================================================
print()
print("After normalization:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)

# ============================================================
# DISPLAY EXAMPLE
# ============================================================
print()
print("Example normalized input:")
print(X_train[0])
print()
print("Corresponding label:")
print(y_train[0])

# ============================================================
# SAVE PREPARED DATA
# ============================================================
print()
print("Saving prepared datasets...")

np.save(config.X_TRAIN_FILE, X_train)
np.save(config.X_VAL_FILE, X_val)
np.save(config.X_TEST_FILE, X_test)
np.save(config.Y_TRAIN_FILE, y_train)
np.save(config.Y_VAL_FILE, y_val)
np.save(config.Y_TEST_FILE, y_test)

# ============================================================
# COMPLETE
# ============================================================
print()
print("==========================================")
print(" DATA PREPARATION COMPLETE")
print("==========================================")
print()
print("Saved files:")
for name in (
    "X_train.npy",
    "X_val.npy",
    "X_test.npy",
    "y_train.npy",
    "y_val.npy",
    "y_test.npy",
    "scaler_mean.npy",
    "scaler_scale.npy",
):
    print(name)
print()
print("==========================================")