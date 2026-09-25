"""GridGuard Fire AI - Step 6: Model evaluation & selection.

  * Reads every experiment's summary (validation metrics) written by step 5.
  * Builds the model comparison table (accuracy / FIRE recall / FIRE
    precision / F1 / float + INT8 size / INT8 accuracy).
  * Performs validation-only threshold analysis on every candidate.
  * Selects the deployment model using a documented, safety-driven rule that
    weights FIRE recall and INT8 robustness above raw accuracy.
  * Computes the FIRE / UNCERTAIN / NO_FIRE thresholds from the validation
    predictions (never the test set).
  * Saves the final float model + final_model_config.json, then produces the
    FIRST (and only) touch of the untouched test set.

Outputs:
  models/keras/gridguard_fire_best.keras
  models/final/final_model_config.json
  results/threshold_report.json
  results/figures/threshold_analysis_<id>.png
  results/figures/architecture_comparison.png
  results/figures/accuracy_vs_size.png
  results/classification_reports/<id>_test_report.txt(+.csv)
  results/confusion_matrices/<id>_confusion_matrix.png
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf  # noqa: E402

from config import (  # noqa: E402
    CACHE_ROOT,
    CLASS_NAMES,
    CLASSIFICATION_REPORTS_DIR,
    COMPARISON_CSV,
    CONFUSION_DIR,
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    FINAL_CONFIG_JSON,
    FINAL_DIR,
    FINAL_MODEL_KERAS,
    RANDOM_SEED,
    SPLITS_DIR,
    THRESHOLD_REPORT_JSON,
)
from _helpers import (  # noqa: E402
    binary_metrics,
    load_manifest,
    predict_probs,
)

# Selection policy -----------------------------------------------------------
# For a fire-safety device we require, on VALIDATION data:
#   * FIRE recall  >= 0.98        (missed fires are the worst failure)
#   * INT8 FIRE recall >= 0.95    (quantization must not destroy recall)
# Among candidates that satisfy both, the smallest INT8 model wins (ESP32).
# Ties break on FIRE F1.
FIRE_RECALL_MIN = 0.98
INT8_FIRE_RECALL_MIN = 0.95
FIRE_STATE_RECALL = 0.98      # threshold chosen so recall >= this
NO_FIRE_SPECIFICITY = 0.99    # no_fire cutoff chosen so specificity >= this

THRESHOLD_GRID = list(np.round(np.arange(0.05, 1.0, 0.01), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-exp", default=None,
                    help="force selection of a specific experiment id")
    args = ap.parse_args()

    print("=== GridGuard Fire AI | Step 6: evaluation & selection ===")
    if not COMPARISON_CSV.exists():
        print("FATAL: no comparison table - run 05_train_models.py first")
        return 1

    experiments = _load_experiments()
    print(f"Experiments found  : {len(experiments)}")

    # ------------------------------ comparison table ----------------------
    _write_comparison_csv(experiments)
    _plot_arch_comparison(experiments)
    _plot_acc_vs_size(experiments)

    # -------------------------- threshold analysis ------------------------
    threshold_rows = {}
    for exp in experiments:
        probs = np.load(EXPERIMENTS_DIR / exp["id"] / "val_probs.npy")
        labels = np.load(EXPERIMENTS_DIR / exp["id"] / "val_labels.npy")
        rows = []
        for t in THRESHOLD_GRID:
            m = binary_metrics(probs, labels, t)
            rows.append({"threshold": t, **{k: m[k] for k in
                          ("accuracy", "fire_recall", "fire_precision", "fire_f1",
                           "fp", "fn")}})
        threshold_rows[exp["id"]] = {"probs": probs, "labels": labels, "rows": rows,
                                     "path": EXPERIMENTS_DIR / exp["id"] / "best.keras",
                                     "size": exp["int8_size"], "exp": exp}
        _plot_threshold_analysis(exp["id"], rows)

    # --------------------------- selection rule ---------------------------
    winner_id = _select(experiments, args.only_exp)
    print(f"\n>>> Selected deployment model: {winner_id}")

    winner_exp = next(e for e in experiments if e["id"] == winner_id)
    info = threshold_rows[winner_id]
    win_probs, win_labels = info["probs"], info["labels"]

    # Fire threshold = highest p that still keeps FIRE recall >= FIRE_STATE_RECALL
    # on validation (highest threshold maximizes FP rejection while satisfying
    # the safety requirement of missing < 2% of real fires).
    fire_thr = 0.5
    cur = 0.0
    for row in info["rows"]:
        if row["fire_recall"] >= FIRE_STATE_RECALL:
            cur = row["threshold"]
    fire_thr = float(cur)
    if fire_thr == 0.0:
        print("WARNING: fire recall target not met; using 0.5")
        fire_thr = 0.5

    # no-fire cutoff = highest p with specificity >= NO_FIRE_SPECIFICITY
    n_fire = int((win_labels == 0).sum())
    no_fire_thr = 0.5
    cur = 0.0
    for row in info["rows"]:
        spec = (n_fire - row["fp"]) / max(1, n_fire)
        if spec >= NO_FIRE_SPECIFICITY:
            cur = row["threshold"]
    no_fire_thr = float(cur)

    state_rows = _state_distribution(win_probs, win_labels, no_fire_thr, fire_thr)

    print(f"  FIRE_THRESHOLD  (>= -> FIRE_DETECTED) : {fire_thr:.2f}")
    print(f"  NO_FIRE_CUTOFF  (<= -> NO_FIRE)       : {no_fire_thr:.2f}")
    print(f"  UNCERTAIN band  ({no_fire_thr:.2f}, {fire_thr:.2f})")
    print("  Validation state distribution:")
    for s, v in state_rows.items():
        print(f"     {s:<14}: {v}")

    # ------------------------------ final float model ---------------------
    shutil.copy2(info["path"], FINAL_MODEL_KERAS)
    model = tf.keras.models.load_model(FINAL_MODEL_KERAS)

    test_manifest = SPLITS_DIR / "test.manifest.csv"
    test_cache = CACHE_ROOT / f"cache_{winner_exp['size']}x{winner_exp['size']}_test"
    test_probs, test_labels = predict_probs(
        model, test_manifest, (winner_exp["size"], winner_exp["size"]),
        32, cache_dir=test_cache)
    np.save(FINAL_DIR / "test_probs.npy", test_probs)
    np.save(FINAL_DIR / "test_labels.npy", test_labels)

    t50 = binary_metrics(test_probs, test_labels, 0.5)
    tfire = binary_metrics(test_probs, test_labels, fire_thr)
    test_report = _report_text(winner_id, winner_exp, t50, tfire, test_probs,
                               test_labels, no_fire_thr, fire_thr)
    (CLASSIFICATION_REPORTS_DIR / f"{winner_id}_test_report.txt").write_text(
        test_report, encoding="utf-8")
    _write_report_csv(winner_id, t50, tfire)
    _plot_confusion(winner_id, test_probs, test_labels, no_fire_thr, fire_thr)
    print(f"\n{test_report}")

    # ------------------------------- persist final config -----------------
    final_cfg = {
        "experiment_id": winner_id,
        "architecture": winner_exp["arch"],
        "input_size": winner_exp["size"],
        "input_channels": 3,
        "seed": RANDOM_SEED,
        "batch_size": winner_exp["batch"],
        "parameters": winner_exp["parameters"],
        "fire_threshold": float(fire_thr),
        "no_fire_cutoff": float(no_fire_thr),
        "uncertain_low": float(no_fire_thr),
        "class_names": CLASS_NAMES,
        "threshold_policy": {
            "fire": f"lowest threshold with validation FIRE recall >= {FIRE_STATE_RECALL}",
            "no_fire": (f"highest threshold with validation no_fire specificity >= "
                        f"{NO_FIRE_SPECIFICITY}"),
        },
        "validation_state_distribution": state_rows,
        "test_metrics_at_fire_threshold": tfire,
        "test_metrics_at_0_5": t50,
    }
    with open(FINAL_CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(final_cfg, fh, indent=2)
    with open(THRESHOLD_REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(final_cfg, fh, indent=2)

    print(f"\nFinal model        : {FINAL_MODEL_KERAS}")
    print(f"Final config       : {FINAL_CONFIG_JSON}")
    return 0


# ---------------------------------------------------------------------------
def _load_experiments():
    from config import BATCH_SIZE

    rows = []
    with open(COMPARISON_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "id": r["experiment"],
                "arch": r["architecture"],
                "size": int(r["input_size"]),
                "params": int(r["parameters"]),
                "batch": BATCH_SIZE,
                "val_acc": float(r["val_accuracy"]),
                "val_recall": float(r["val_fire_recall"]),
                "val_precision": float(r["val_fire_precision"]),
                "val_f1": float(r["val_fire_f1"]),
                "val_fn": int(r["val_fn"]),
                "val_fp": int(r["val_fp"]),
                "int8_acc": _safe_float(r.get("int8_val_accuracy")),
                "int8_recall": _safe_float(r.get("int8_val_fire_recall")),
                "float_size": int(r["float_tflite_size_bytes"]),
                "int8_size": int(r["int8_tflite_size_bytes"]),
            })
    return rows


def _safe_float(v):
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return None


def _write_comparison_csv(experiments):
    path = EXPERIMENTS_DIR / "model_comparison_validation.csv"
    fieldnames = ["experiment", "architecture", "input_size", "parameters",
                  "val_accuracy", "val_fire_recall", "val_fire_precision",
                  "val_fire_f1", "val_fn", "val_fp",
                  "int8_val_accuracy", "int8_val_fire_recall",
                  "float_tflite_size_bytes", "int8_tflite_size_bytes"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for e in experiments:
            w.writerow({
                "experiment": e["id"], "architecture": e["arch"],
                "input_size": e["size"], "parameters": e["params"],
                "val_accuracy": e["val_acc"], "val_fire_recall": e["val_recall"],
                "val_fire_precision": e["val_precision"], "val_fire_f1": e["val_f1"],
                "val_fn": e["val_fn"], "val_fp": e["val_fp"],
                "int8_val_accuracy": e["int8_acc"], "int8_val_fire_recall": e["int8_recall"],
                "float_tflite_size_bytes": e["float_size"], "int8_tflite_size_bytes": e["int8_size"],
            })


def _select(experiments, forced=None):
    if forced:
        return forced
    candidates = []
    misses = []
    for e in experiments:
        recall_ok = e["val_recall"] >= FIRE_RECALL_MIN
        int8_ok = (e["int8_recall"] is not None and e["int8_recall"] >= INT8_FIRE_RECALL_MIN)
        if recall_ok and int8_ok:
            candidates.append(e)
        else:
            misses.append((e["id"], e["val_recall"], e["int8_recall"]))
    print("\nCandidates passing (recall>=%.2f) & (int8 recall>=%.2f):"
          % (FIRE_RECALL_MIN, INT8_FIRE_RECALL_MIN))
    for c in sorted(candidates, key=lambda x: x["int8_size"]):
        print(f"  {c['id']:<22} int8={c['int8_size']:>6}B f1={c['val_f1']:.3f}")
    for m in misses:
        print(f"  rejected {m[0]:<22} val_recall={m[1]} int8_recall={m[2]}")
    if not candidates:
        print("WARNING: no candidate met both recall targets - using rule "
              "fallback (best combined score).")
        return max(experiments, key=lambda e: (e["int8_recall"] or 0) * e["val_f1"])["id"]
    # smallest INT8 model first, tie-break by FIRE F1
    cand = sorted(candidates, key=lambda x: (x["int8_size"], -x["val_f1"]))
    return cand[0]["id"]


def _plot_threshold_analysis(exp_id, rows):
    t = [r["threshold"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, [r["fire_recall"] for r in rows], label="FIRE recall")
    axes[0].plot(t, [r["fire_precision"] for r in rows], label="FIRE precision")
    axes[0].plot(t, [r["fire_f1"] for r in rows], label="FIRE F1")
    axes[0].plot(t, [r["accuracy"] for r in rows], label="accuracy")
    axes[0].axhline(0.98, ls="--", color="gray", lw=0.8)
    axes[0].legend(); axes[0].set_ylabel("metric"); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Threshold analysis - {exp_id}")
    axes[1].plot(t, [r["fp"] for r in rows], label="false positives")
    axes[1].plot(t, [r["fn"] for r in rows], label="false negatives")
    axes[1].set_xlabel("FIRE threshold"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"threshold_analysis_{exp_id}.png", dpi=110)
    plt.close(fig)


def _plot_arch_comparison(experiments):
    ids = [e["id"] for e in experiments]
    recall = [e["val_recall"] for e in experiments]
    int8_recall = [e["int8_recall"] if e["int8_recall"] else 0 for e in experiments]
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, recall, 0.38, label="FLOAT FIRE recall (val)")
    ax.bar(x + 0.2, int8_recall, 0.38, label="INT8 FIRE recall (val)")
    ax.axhline(0.98, ls="--", color="red", lw=0.8, label="0.98 target")
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=30, ha="right")
    ax.set_ylabel("recall"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("FIRE recall by candidate (float vs INT8)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "architecture_comparison.png", dpi=110)
    plt.close(fig)


def _plot_acc_vs_size(experiments):
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in experiments:
        ax.scatter(e["int8_size"], e["val_acc"], s=90, label=e["id"])
        ax.annotate(e["id"], (e["int8_size"], e["val_acc"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("INT8 model size (bytes)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Accuracy vs INT8 size trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_vs_size.png", dpi=110)
    plt.close(fig)


def _state_distribution(probs, labels, no_fire_thr, fire_thr):
    states = {"NO_FIRE": 0, "UNCERTAIN": 0, "FIRE_DETECTED": 0}
    states_true = {"NO_FIRE": 0, "UNCERTAIN": 0, "FIRE_DETECTED": 0}
    for p, lbl in zip(probs, labels):
        if p >= fire_thr:
            states["FIRE_DETECTED"] += 1
            states_true["FIRE_DETECTED"] += int(lbl == 1)
        elif p > no_fire_thr:
            states["UNCERTAIN"] += 1
            states_true["UNCERTAIN"] += int(lbl == 1)
        else:
            states["NO_FIRE"] += 1
            states_true["NO_FIRE"] += int(lbl == 1)
    return {"counts": states, "true_fire_within_state": states_true}


def _report_text(winner_id, e, t50, tfire, probs, labels, no_fire_thr, fire_thr):
    from sklearn.metrics import classification_report, confusion_matrix

    preds50 = (probs >= 0.5).astype(int)
    preds_state = np.where(probs >= fire_thr, 1, np.where(probs > no_fire_thr, -1, 0))
    cm = confusion_matrix(labels, preds50)
    return f"""GRIDGUARD FIRE AI - FINAL FLOAT MODEL TEST REPORT
Winner         : {winner_id}
Architecture   : {e['arch']}
Input size     : {e['size']} x {e['size']} x 3
Parameters     : {e['params']}
Test images    : {len(probs)} ({int((labels == 1).sum())} fire / {int((labels == 0).sum())} no_fire)

Thresholds (from validation):
  FIRE_DETECTED  : p >= {fire_thr:.2f}
  UNCERTAIN      : {no_fire_thr:.2f} < p < {fire_thr:.2f}
  NO_FIRE        : p <= {no_fire_thr:.2f}

Metrics at 0.5 threshold          : acc={t50['accuracy']} fire_recall={t50['fire_recall']} fire_precision={t50['fire_precision']} fire_f1={t50['fire_f1']} (fn={t50['fn']}, fp={t50['fp']})
Metrics at FIRE_THRESHOLD={fire_thr:.2f} : acc={tfire['accuracy']} fire_recall={tfire['fire_recall']} fire_precision={tfire['fire_precision']} fire_f1={tfire['fire_f1']} (fn={tfire['fn']}, fp={tfire['fp']})

Classification report (at 0.5):
{classification_report(labels, preds50, target_names=CLASS_NAMES)}

Confusion matrix (at 0.5, rows=truth, cols=pred):
  {cm} == [[TN, FP], [FN, TP]]
State mapping counts (over test):
  {_state_distribution(probs, labels, no_fire_thr, fire_thr)}
    """


def _write_report_csv(winner_id, t50, tfire):
    with open(CLASSIFICATION_REPORTS_DIR / f"{winner_id}_test_metrics.csv",
              "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["setting", "accuracy", "fire_recall",
                           "fire_precision", "fire_f1", "tp", "tn",
                           "fn", "fp", "threshold"])
        w.writeheader()
        w.writerow({"setting": "threshold_0.5", **t50})
        w.writerow({"setting": "threshold_fire", **tfire})


def _plot_confusion(winner_id, probs, labels, no_fire_thr, fire_thr):
    from sklearn.metrics import confusion_matrix

    preds_fire = (probs >= fire_thr).astype(int)
    cm = confusion_matrix(labels, preds_fire)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im_ = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["NO_FIRE", "FIRE"]); ax.set_yticklabels(["NO_FIRE", "FIRE"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Test confusion matrix @ p>={fire_thr:.2f}\n{winner_id}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im_, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_DIR / f"{winner_id}_confusion_matrix.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())