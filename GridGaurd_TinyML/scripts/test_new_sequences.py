import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI
# MODEL 1 - UNSEEN DATA TEST
# Temperature Anomaly Detection
# ============================================================
print("==========================================")
print(" GRIDGUARD AI - MODEL 1")
print(" UNSEEN DATA TEST")
print("==========================================")

MODEL_PATH = config.KERAS_MODEL_FILE
WINDOW_SIZE = config.WINDOW_SIZE
ANOMALY_THRESHOLD = config.ANOMALY_THRESHOLD


# ============================================================
# LOAD MODEL
# ============================================================
print()
print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD EXACT TRAINING NORMALIZATION
# ============================================================
print()
print("Loading scaler parameters...")

scaler_mean = np.load(config.SCALER_MEAN_FILE)
scaler_scale = np.load(config.SCALER_SCALE_FILE)

print("Scaler loaded successfully.")

# ============================================================
# NEW UNSEEN TEMPERATURE SEQUENCES
# ============================================================
scenarios = {
    "Normal Stable": np.array([
        29.0, 29.1, 29.1, 29.2, 29.1,
        29.2, 29.2, 29.3, 29.2, 29.2,
        29.3, 29.3, 29.2, 29.3, 29.3,
        29.4, 29.3, 29.3, 29.4, 29.3,
    ]),
    "Normal Slight Drift": np.array([
        28.5, 28.6, 28.6, 28.7, 28.7,
        28.8, 28.8, 28.9, 28.9, 29.0,
        29.0, 29.0, 29.1, 29.1, 29.1,
        29.2, 29.2, 29.2, 29.3, 29.3,
    ]),
    "Rapid Temperature Rise": np.array([
        29.0, 29.1, 29.2, 29.3, 29.5,
        29.8, 30.1, 30.5, 30.9, 31.3,
        31.7, 32.1, 32.6, 33.0, 33.5,
        34.0, 34.5, 35.0, 35.5, 36.0,
    ]),
    "Sudden Thermal Spike": np.array([
        29.0, 29.1, 29.0, 29.1, 29.1,
        29.0, 29.1, 29.1, 29.0, 29.1,
        29.0, 29.1, 29.1, 29.2, 29.2,
        31.0, 32.0, 33.0, 34.0, 35.0,
    ]),
    "Unstable Temperature": np.array([
        29.0, 29.8, 28.9, 30.0, 28.5,
        30.2, 28.7, 30.4, 28.6, 30.1,
        28.4, 30.5, 28.8, 30.3, 28.5,
        30.6, 28.7, 30.4, 28.6, 30.5,
    ]),
}

# ============================================================
# TEST
# ============================================================
print()
print("==========================================")
print(" TESTING NEW TEMPERATURE SEQUENCES")
print("==========================================")

for name, temperatures in scenarios.items():
    if len(temperatures) != WINDOW_SIZE:
        print(f"ERROR: {name} does not contain {WINDOW_SIZE} samples.")
        continue

    # Apply the exact same normalization used in training.
    normalized = (temperatures - scaler_mean) / scaler_scale

    model_input = normalized.reshape(1, WINDOW_SIZE)

    probability = model.predict(model_input, verbose=0)[0][0]

    result = "ANOMALY" if probability >= ANOMALY_THRESHOLD else "NORMAL"

    print()
    print("------------------------------------------")
    print(name)
    print("------------------------------------------")
    print(f"Temperature range : {temperatures.min():.1f} - {temperatures.max():.1f} °C")
    print(f"Anomaly probability: {probability * 100:.2f}%")
    print(f"Threshold          : {ANOMALY_THRESHOLD}")
    print(f"Decision            : {result}")

# ============================================================
# COMPLETE
# ============================================================
print()
print("==========================================")
print(" UNSEEN DATA TEST COMPLETE")
print("==========================================")