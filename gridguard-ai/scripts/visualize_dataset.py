import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI
# DATASET VISUALIZATION
# ============================================================

DATA_FILE = config.DATASET_FILE
WINDOW_SIZE = config.WINDOW_SIZE


# ============================================================
# LOAD DATASET
# ============================================================
df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print()
print("Dataset shape:")
print(df.shape)
print()
print("Class distribution:")
print(df["label"].value_counts())
print()
print("First 5 rows:")
print(df.head())

# ============================================================
# TEMPERATURE COLUMNS
# ============================================================
temperature_columns = [f"temp_{i}" for i in range(1, WINDOW_SIZE + 1)]

normal_samples = df[df["label"] == 0].head(5)
anomaly_samples = df[df["label"] == 1].head(5)

# ============================================================
# PLOT NORMAL SAMPLES
# ============================================================
plt.figure(figsize=(10, 6))

for _, row in normal_samples.iterrows():
    plt.plot(range(1, WINDOW_SIZE + 1), row[temperature_columns].values, marker="o")

plt.title("GridGuard AI - Normal Temperature Patterns")
plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT ANOMALY SAMPLES
# ============================================================
plt.figure(figsize=(10, 6))

for _, row in anomaly_samples.iterrows():
    plt.plot(range(1, WINDOW_SIZE + 1), row[temperature_columns].values, marker="o")

plt.title("GridGuard AI - Anomalous Temperature Patterns")
plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

print()
print("Visualization complete!")