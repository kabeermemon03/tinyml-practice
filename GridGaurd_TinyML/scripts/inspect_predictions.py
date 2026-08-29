import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI - PREDICTION INSPECTION
# ============================================================
print("=" * 60)
print(" GRIDGUARD AI - MODEL 1 PREDICTION INSPECTION")
print("=" * 60)

MODEL_PATH = config.KERAS_MODEL_FILE


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

X_test = np.load(config.X_TEST_FILE)
y_test = np.load(config.Y_TEST_FILE)

scaler_mean = np.load(config.SCALER_MEAN_FILE)
scaler_scale = np.load(config.SCALER_SCALE_FILE)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------
probabilities = model.predict(X_test, verbose=0).flatten()

# Experimental detection threshold used throughout validation.
threshold = 0.1

y_pred = (probabilities >= threshold).astype(int)

# ------------------------------------------------------------
# Find misclassified samples
# ------------------------------------------------------------
wrong = np.where(y_test != y_pred)[0]

print("\nUsing threshold:", threshold)
print("Misclassified samples:", len(wrong))

# ------------------------------------------------------------
# Convert normalized data back to °C
# ------------------------------------------------------------
X_temperature = X_test * scaler_scale + scaler_mean

# ------------------------------------------------------------
# Print missed anomalies
# ------------------------------------------------------------
print("\n" + "=" * 60)
print(" MISSED ANOMALIES")
print("=" * 60)

missed = np.where((y_test == 1) & (y_pred == 0))[0]

for i in missed:
    print("\n" + "-" * 60)
    print(f"Test index : {i}")
    print("Actual     : ANOMALY")
    print("Predicted  : NORMAL")
    print(f"Probability: {probabilities[i]:.6f}")

    print("\nTemperature (°C):")
    temps = X_temperature[i]

    for j, temp in enumerate(temps):
        print(f"  Sample {j + 1:02d}: {temp:.2f} °C")

    print(f"\nMinimum: {temps.min():.2f} °C")
    print(f"Maximum: {temps.max():.2f} °C")
    print(f"Rise: {temps.max() - temps.min():.2f} °C")

# ------------------------------------------------------------
# Correct anomaly examples
# ------------------------------------------------------------
print("\n" + "=" * 60)
print(" CORRECTLY DETECTED ANOMALIES")
print("=" * 60)

correct_anomalies = np.where((y_test == 1) & (y_pred == 1))[0]

# Show first 5
for i in correct_anomalies[:5]:
    temps = X_temperature[i]

    print("\n" + "-" * 60)
    print(f"Test index : {i}")
    print("Actual     : ANOMALY")
    print("Predicted  : ANOMALY")
    print(f"Probability: {probabilities[i]:.6f}")
    print("Temperature:", np.round(temps, 2))
    print(f"Min: {temps.min():.2f} °C | Max: {temps.max():.2f} °C | Range: {temps.max() - temps.min():.2f} °C")

# ------------------------------------------------------------
# Normal examples
# ------------------------------------------------------------
print("\n" + "=" * 60)
print(" NORMAL EXAMPLES")
print("=" * 60)

normal_indices = np.where(y_test == 0)[0]

for i in normal_indices[:5]:
    temps = X_temperature[i]

    print("\n" + "-" * 60)
    print(f"Test index : {i}")
    print("Actual     : NORMAL")
    print("Predicted  : NORMAL")
    print(f"Probability: {probabilities[i]:.6f}")
    print("Temperature:", np.round(temps, 2))
    print(f"Min: {temps.min():.2f} °C | Max: {temps.max():.2f} °C | Range: {temps.max() - temps.min():.2f} °C")

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
print("\n" + "=" * 60)
print(" INSPECTION COMPLETE")
print("=" * 60)