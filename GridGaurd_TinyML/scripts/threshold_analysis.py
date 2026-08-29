import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI
# MODEL 1 - THRESHOLD ANALYSIS
# ============================================================
print("==========================================")
print(" GRIDGUARD AI - MODEL 1")
print(" THRESHOLD ANALYSIS")
print("==========================================")

# ============================================================
# LOAD DATA
# ============================================================
y_test = np.load(config.Y_TEST_FILE)
probabilities = np.load(config.TEST_PROBABILITIES_FILE)

# ============================================================
# THRESHOLDS TO TEST
# ============================================================
thresholds = [0.50, 0.45, 0.40, 0.35, 0.30]

# ============================================================
# HEADER
# ============================================================
print()
print("Testing different anomaly thresholds...")
print()
print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1 Score':<12}"
)
print("-" * 60)

# ============================================================
# ANALYZE EACH THRESHOLD
# ============================================================
results = []

for threshold in thresholds:
    y_pred = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append((threshold, accuracy, precision, recall, f1))

    print(
        f"{threshold:<12.2f}"
        f"{accuracy * 100:<12.2f}"
        f"{precision * 100:<12.2f}"
        f"{recall * 100:<12.2f}"
        f"{f1 * 100:<12.2f}"
    )

# ============================================================
# CONFUSION MATRICES
# ============================================================
print()
print("==========================================")
print(" CONFUSION MATRICES")
print("==========================================")

for threshold in thresholds:
    y_pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print()
    print(f"Threshold: {threshold:.2f}")
    print("                 Predicted")
    print("              Normal  Anomaly")
    print(f"Actual Normal   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual Anomaly  {cm[1][0]:4d}    {cm[1][1]:4d}")

# ============================================================
# BEST F1
# ============================================================
best = max(results, key=lambda x: x[4])

print()
print("==========================================")
print(" BEST THRESHOLD")
print("==========================================")
print()
print(f"Threshold : {best[0]:.2f}")
print(f"Accuracy  : {best[1] * 100:.2f}%")
print(f"Precision : {best[2] * 100:.2f}%")
print(f"Recall    : {best[3] * 100:.2f}%")
print(f"F1 Score  : {best[4] * 100:.2f}%")

print()
print("==========================================")
print(" ANALYSIS COMPLETE")
print("==========================================")