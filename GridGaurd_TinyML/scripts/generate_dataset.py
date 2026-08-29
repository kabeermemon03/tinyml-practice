import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI
# MODEL 1 - SYNTHETIC TEMPERATURE DATASET GENERATOR
# Temperature Anomaly Detection
# ============================================================

NUM_NORMAL = 2500
NUM_ANOMALY = 2500

WINDOW_SIZE = config.WINDOW_SIZE
SAMPLING_TIME = config.SAMPLING_TIME
BASE_TEMPERATURE = config.BASE_TEMPERATURE
RANDOM_SEED = config.RANDOM_SEED

np.random.seed(RANDOM_SEED)


# ============================================================
# NORMAL TEMPERATURE
# ============================================================
def generate_normal_window():
    temperature = BASE_TEMPERATURE + np.random.uniform(-1.5, 1.5)
    drift = np.random.uniform(-0.03, 0.03)
    values = []

    for _ in range(WINDOW_SIZE):
        temperature += drift + np.random.normal(0, 0.08)
        values.append(temperature)

    return values


# ============================================================
# ANOMALY TEMPERATURE
# ============================================================
ANOMALY_TYPES = [
    "rapid_rise",
    "gradual_rise",
    "sudden_spike",
    "fluctuation",
    "extreme_fluctuation",
    "oscillation",
]


def generate_anomaly_window():
    temperature = BASE_TEMPERATURE + np.random.uniform(-1.5, 1.5)
    anomaly_type = np.random.choice(ANOMALY_TYPES)
    values = []

    for i in range(WINDOW_SIZE):
        noise = np.random.normal(0, 0.08)

        if anomaly_type == "rapid_rise":
            temperature += np.random.uniform(0.25, 0.55)
        elif anomaly_type == "gradual_rise":
            temperature += np.random.uniform(0.12, 0.25)
        elif anomaly_type == "sudden_spike":
            if i < 12:
                temperature += np.random.uniform(-0.03, 0.03)
            else:
                temperature += np.random.uniform(0.8, 1.5)
        elif anomaly_type == "fluctuation":
            temperature += np.random.uniform(-0.8, 0.8)
        elif anomaly_type == "extreme_fluctuation":
            temperature += np.random.uniform(0.6, 1.2) if i % 2 == 0 else -np.random.uniform(0.6, 1.2)
        elif anomaly_type == "oscillation":
            temperature += 1.0 * np.sin(i * np.pi / 2)

        temperature += noise
        values.append(temperature)

    return values


# ============================================================
# CREATE DATASET
# ============================================================
print("Generating NORMAL samples...")
dataset = [generate_normal_window() + [0] for _ in range(NUM_NORMAL)]

print("Generating ANOMALY samples...")
dataset += [generate_anomaly_window() + [1] for _ in range(NUM_ANOMALY)]

columns = [f"temp_{i + 1}" for i in range(WINDOW_SIZE)] + ["label"]

df = pd.DataFrame(dataset, columns=columns)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# ============================================================
# SAVE DATASET
# ============================================================
OUTPUT_FILE = config.DATASET_FILE
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# SUMMARY
# ============================================================
print()
print("==========================================")
print(" GRIDGUARD AI DATASET CREATED")
print("==========================================")
print(f"Total samples : {len(df)}")
print(f"Normal        : {(df['label'] == 0).sum()}")
print(f"Anomaly       : {(df['label'] == 1).sum()}")
print(f"Window size   : {WINDOW_SIZE}")
print(f"Sampling      : {SAMPLING_TIME} second")
print()
print("Anomaly types:")
for name in ANOMALY_TYPES:
    print(f" - {name.title()}")
print()
print("Saved to:")
print(OUTPUT_FILE)
print("==========================================")

# ============================================================
# DATASET INSPECTION
# ============================================================
print("\n==========================================")
print(" DATASET INSPECTION")
print("==========================================")
print("\nFirst 5 samples:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nMissing values:")
print(df.isnull().sum().sum())
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nTemperature statistics:")
print(df.drop(columns=["label"]).describe())