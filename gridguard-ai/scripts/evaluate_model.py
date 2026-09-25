import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
# GRIDGUARD AI
# MODEL 1 - FINAL EVALUATION
# ============================================================

MODEL_PATH = config.KERAS_MODEL_FILE


# ============================================================
# LOAD MODEL
# ============================================================
print("==========================================")
print(" GRIDGUARD AI - MODEL 1 EVALUATION")
print("==========================================")

print()
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD TEST DATA
# ============================================================
X_test = np.load(config.X_TEST_FILE)
y_test = np.load(config.Y_TEST_FILE)

print()
print("Test samples:", len(X_test))

# ============================================================
# MODEL PREDICTIONS
# ============================================================
print()
print("Generating predictions...")

probabilities = model.predict(X_test, verbose=0).flatten()

# ============================================================
# CONVERT PROBABILITY TO CLASS
# ============================================================
threshold = config.DEFAULT_THRESHOLD
y_pred = (probabilities >= threshold).astype(int)

# ============================================================
# ACCURACY
# ============================================================
accuracy = accuracy_score(y_test, y_pred)

print()
print("==========================================")
print(" PERFORMANCE")
print("==========================================")
print()
print(f"Accuracy: {accuracy * 100:.2f}%")

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)

print()
print("==========================================")
print(" CONFUSION MATRIX")
print("==========================================")
print()
print("                 Predicted")
print("              Normal  Anomaly")
print(f"Actual Normal   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"Actual Anomaly  {cm[1][0]:4d}    {cm[1][1]:4d}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print()
print("==========================================")
print(" CLASSIFICATION REPORT")
print("==========================================")
print()
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Anomaly"],
        digits=4,
    )
)

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================
plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("GridGuard AI - Model 1 Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks([0, 1], ["Normal", "Anomaly"])
plt.yticks([0, 1], ["Normal", "Anomaly"])

# Display values inside matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()

confusion_path = config.CONFUSION_MATRIX_FILE
plt.savefig(confusion_path, dpi=200)
plt.show()

# ============================================================
# SAVE PREDICTIONS
# ============================================================
np.save(config.TEST_PROBABILITIES_FILE, probabilities)
np.save(config.TEST_PREDICTIONS_FILE, y_pred)

# ============================================================
# COMPLETE
# ============================================================
print()
print("==========================================")
print(" EVALUATION COMPLETE")
print("==========================================")
print()
print("Confusion matrix saved to:")
print(confusion_path)
print()
print("Model evaluation files saved in:")
print(config.MODEL_DIR)