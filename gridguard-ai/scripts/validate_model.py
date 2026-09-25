import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI - MODEL 1 VALIDATION
# ============================================================
print("=" * 50)
print(" GRIDGUARD AI - MODEL 1 VALIDATION")
print("=" * 50)

MODEL_PATH = config.KERAS_MODEL_FILE


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------
print("\nLoading test data...")

X_test = np.load(config.X_TEST_FILE)
y_test = np.load(config.Y_TEST_FILE)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ------------------------------------------------------------
# Load scaler
# ------------------------------------------------------------
scaler_mean = np.load(config.SCALER_MEAN_FILE)
scaler_scale = np.load(config.SCALER_SCALE_FILE)

print("\nScaler:")
print(f"Mean : {scaler_mean}")
print(f"Scale: {scaler_scale}")

# ------------------------------------------------------------
# Basic dataset checks
# ------------------------------------------------------------
print("\n" + "=" * 50)
print(" DATASET CHECK")
print("=" * 50)

print(f"Total test samples : {len(X_test)}")

unique, counts = np.unique(y_test, return_counts=True)

for label, count in zip(unique, counts):
    name = "Normal" if label == 0 else "Anomaly"
    print(f"{name:10s}: {count}")

print("\nNaN check:")
print("X_test NaN:", np.isnan(X_test).any())
print("y_test NaN:", np.isnan(y_test).any())
print("\nInfinite value check:")
print("X_test Inf:", np.isinf(X_test).any())

# ------------------------------------------------------------
# Model predictions
# ------------------------------------------------------------
print("\n" + "=" * 50)
print(" MODEL PREDICTIONS")
print("=" * 50)

probabilities = model.predict(X_test, verbose=0).flatten()

threshold = config.DEFAULT_THRESHOLD
y_pred = (probabilities >= threshold).astype(int)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print(f"\nThreshold : {threshold}")
print(f"Accuracy  : {accuracy * 100:.2f}%")

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Anomaly"],
    )
)

# ------------------------------------------------------------
# Probability statistics
# ------------------------------------------------------------
print("=" * 50)
print(" PROBABILITY ANALYSIS")
print("=" * 50)

normal_probs = probabilities[y_test == 0]
anomaly_probs = probabilities[y_test == 1]

print("\nNormal samples:")
print(f"Min : {normal_probs.min():.4f}")
print(f"Max : {normal_probs.max():.4f}")
print(f"Mean: {normal_probs.mean():.4f}")

print("\nAnomaly samples:")
print(f"Min : {anomaly_probs.min():.4f}")
print(f"Max : {anomaly_probs.max():.4f}")
print(f"Mean: {anomaly_probs.mean():.4f}")

# ------------------------------------------------------------
# Threshold analysis
# ------------------------------------------------------------
print("\n" + "=" * 50)
print(" THRESHOLD ANALYSIS")
print("=" * 50)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Normal FP':<12}"
    f"{'Anomaly FN':<12}"
)

for threshold in np.arange(0.1, 1.0, 0.1):
    pred = (probabilities >= threshold).astype(int)

    cm_t = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm_t.ravel()
    acc = accuracy_score(y_test, pred)

    print(
        f"{threshold:<12.1f}"
        f"{acc * 100:<12.2f}"
        f"{fp:<12}"
        f"{fn:<12}"
    )

# ------------------------------------------------------------
# Misclassified samples
# ------------------------------------------------------------
print("\n" + "=" * 50)
print(" MISCLASSIFIED SAMPLES")
print("=" * 50)

wrong = np.where(y_test != y_pred)[0]

print(f"\nTotal misclassified: {len(wrong)}")

for i in wrong[:20]:
    actual = "Normal" if y_test[i] == 0 else "Anomaly"
    predicted = "Normal" if y_pred[i] == 0 else "Anomaly"

    print(
        f"Index {i:4d} | "
        f"Actual: {actual:7s} | "
        f"Predicted: {predicted:7s} | "
        f"Probability: {probabilities[i]:.4f}"
    )

# ------------------------------------------------------------
# Sample predictions
# ------------------------------------------------------------
print("\n" + "=" * 50)
print(" SAMPLE PREDICTIONS")
print("=" * 50)

for i in range(min(10, len(X_test))):
    actual = "Normal" if y_test[i] == 0 else "Anomaly"
    predicted = "Normal" if y_pred[i] == 0 else "Anomaly"

    print(
        f"\nSample {i + 1}"
        f"\nActual     : {actual}"
        f"\nPredicted  : {predicted}"
        f"\nProbability: {probabilities[i]:.4f}"
    )

    print("Temperature window:")
    print(np.round(X_test[i], 3))

print("\n" + "=" * 50)
print(" MODEL 1 VALIDATION COMPLETE")
print("=" * 50)