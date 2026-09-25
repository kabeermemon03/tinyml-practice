"""Step 6: GridGuard AI v2 - full evaluation.

Test accuracy, per-class precision/recall/F1, confusion matrix and a
dedicated analysis of the tricky pairs:
  TEMPERATURE_RISING vs THERMAL_DRIFT
  THERMAL_DRIFT vs RAPID_TEMPERATURE_RISE
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_CHECKPOINT,
    CLASS_NAMES,
    CLASS_NAMES_LIST,
    CONFUSION_PNG,
    PAIR_ANALYSIS_CSV,
    REPORT_CSV,
    TEST_NPZ,
)


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 6: Test-set evaluation + confusion analysis")
    print("=" * 72)
    d = np.load(TEST_NPZ)
    X_test, y_test = d["X"], d["y"]
    model = tf.keras.models.load_model(BEST_CHECKPOINT)
    print(f"Loaded model: {BEST_CHECKPOINT}")

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.4f}")

    print("\nPer-class classification report:")
    report = classification_report(
        y_test, preds, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES_LIST, output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_test, preds, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES_LIST, zero_division=0,
    ))

    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for c in sorted(CLASS_NAMES):
            ref = report[CLASS_NAMES[c]]
            writer.writerow([CLASS_NAMES[c], round(ref["precision"], 4),
                             round(ref["recall"], 4), round(ref["f1-score"], 4),
                             int(ref["support"])])
    print(f"Classification report saved: {REPORT_CSV}")

    cm = confusion_matrix(y_test, preds, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES_LIST, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES_LIST)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=130)
    plt.close(fig)
    print(f"Confusion matrix: {CONFUSION_PNG}")

    print("\nPair confusion analysis:")
    pairs = [(1, 2), (2, 3), (1, 3), (0, 1)]
    pair_rows = []
    for a, b in pairs:
        mask = (y_test == a) | (y_test == b)
        sub_cm = confusion_matrix(y_test[mask], preds[mask], labels=[a, b])
        both = sub_cm[0][1] + sub_cm[1][0]
        total = int(mask.sum())
        pct = 100.0 * both / total if total else 0.0
        print(f"  {CLASS_NAMES[a]} <-> {CLASS_NAMES[b]}: {both}/{total} "
              f"({pct:.1f}%) confused either way")
        pair_rows.append([CLASS_NAMES[a], CLASS_NAMES[b], both, total, round(pct, 2)])
    with open(PAIR_ANALYSIS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_a", "class_b", "mutual_confusions", "total_instances", "mutual_confusion_pct"])
        writer.writerows(pair_rows)
    print(f"Pair analysis saved: {PAIR_ANALYSIS_CSV}")


if __name__ == "__main__":
    main()