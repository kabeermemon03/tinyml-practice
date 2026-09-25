"""GridGuard Fire AI - Step 10: Quantized (INT8) model evaluation.

Compares float32 vs INT8 performance on the untouched test set and produces
the official float-vs-int8 report + figure.  Also recomputes the state
distribution (NO_FIRE / UNCERTAIN / FIRE_DETECTED) with the deployed INT8
probabilities and the selected thresholds.

Outputs:
  results/float_vs_int8_comparison.csv
  results/figures/float_vs_int8_accuracy.png
  results/quantized_evaluation_report.txt
"""
from __future__ import annotations

import csv
import json
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
    FLOAT_VS_INT8_CSV,
    FINAL_CONFIG_JSON,
    FINAL_DIR,
    INT8_TFLITE,
    QUANT_PARAMS_JSON,
    RESULTS_DIR,
    SPLITS_DIR,
)
from _helpers import (  # noqa: E402
    binary_metrics,
    build_tf_dataset,
)

REPORT_TXT = RESULTS_DIR / "quantized_evaluation_report.txt"
FIGURE = RESULTS_DIR / "figures" / "float_vs_int8_accuracy.png"


def main():
    print("=== GridGuard Fire AI | Step 10: INT8 evaluation ===")
    cfg = json.loads(FINAL_CONFIG_JSON.read_text(encoding="utf-8"))
    params = json.loads(QUANT_PARAMS_JSON.read_text(encoding="utf-8"))
    size = cfg["input_size"]
    fire_thr = float(cfg["fire_threshold"])

    float_probs = np.load(FINAL_DIR / "float32_tflite_test_probs.npy")
    int8_probs, test_labels = _run_int8(INT8_TFLITE, size)

    rows = []
    for label, probs in (("float32", float_probs), ("int8", int8_probs)):
        for tname, thr in (("0.5", 0.5), ("fire_thr", fire_thr)):
            m = binary_metrics(probs, test_labels, thr)
            rows.append({"model": label, "threshold": tname, **m})

    with open(FLOAT_VS_INT8_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "threshold", "accuracy",
                                           "fire_recall", "fire_precision",
                                           "fire_f1", "tp", "tn", "fp", "fn"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['model']:<8} @ {r['threshold']:<8} acc={r['accuracy']} "
              f"recall={r['fire_recall']} prec={r['fire_precision']} "
              f"f1={r['fire_f1']} fn={r['fn']} fp={r['fp']}")

    _plot(float_probs, int8_probs, test_labels)

    # state distribution with INT8 probs + thresholds
    states = _state_distribution(int8_probs, test_labels, cfg["no_fire_cutoff"], fire_thr)
    report = [
        "=" * 70,
        "GRIDGUARD FIRE AI - QUANTIZED MODEL EVALUATION",
        "=" * 70,
        f"Input size        : {params['input_size']}x{params['input_size']}x3",
        f"INT8 size         : {params['size_bytes']} bytes",
        f"Input scale/zp    : {params['input']['scale']} / {params['input']['zero_point']}",
        f"Output scale/zp   : {params['output']['scale']} / {params['output']['zero_point']}",
        f"Fire threshold    : {fire_thr}",
        "",
        "  MODEL        THRESHOLD  ACC  RECALL  PREC   F1    FN  FP",
    ]
    for r in rows:
        report.append(
            f"  {r['model']:<12}{r['threshold']:<10}{r['accuracy']:.3f} "
            f"{r['fire_recall']:.3f}  {r['fire_precision']:.3f}  "
            f"{r['fire_f1']:.3f}  {r['fn']:>3} {r['fp']:>3}")
    report += [
        "",
        "State distribution on test with INT8 model (deployed):",
        f"  NO_FIRE       : {states['NO_FIRE']}",
        f"  UNCERTAIN     : {states['UNCERTAIN']}",
        f"  FIRE_DETECTED : {states['FIRE_DETECTED']}",
        "",
        "True-fire images in each state:",
        f"  {states}",
        "=" * 70,
    ]
    REPORT_TXT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


def _run_int8(tflite_bytes_path, size):
    from config import CACHE_ROOT

    test_cache = CACHE_ROOT / f"cache_{size}x{size}_test"
    ds = build_tf_dataset(SPLITS_DIR / "test.manifest.csv", (size, size),
                          32, train=False, cache_dir=test_cache)
    interp = tf.lite.Interpreter(model_path=str(tflite_bytes_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = in_d["quantization"]
    out_scale, out_zp = out_d["quantization"]

    probs, labels = [], []
    for xb, yb in ds:
        for xi, yi in zip(xb.numpy(), yb.numpy()):
            q = np.round(xi / in_scale + in_zp).astype(np.int8)
            interp.set_tensor(in_d["index"], q[None, ...])
            interp.invoke()
            out = interp.get_tensor(out_d["index"])[0, 0]
            probs.append((out.astype(np.float32) - out_zp) * out_scale)
            labels.append(int(yi))
    return np.asarray(probs), np.asarray(labels)


def _state_distribution(probs, labels, no_fire_thr, fire_thr):
    out = {"NO_FIRE": {"count": 0, "true_fire": 0},
           "UNCERTAIN": {"count": 0, "true_fire": 0},
           "FIRE_DETECTED": {"count": 0, "true_fire": 0}}
    for p, lbl in zip(probs, labels):
        if p >= fire_thr:
            out["FIRE_DETECTED"]["count"] += 1
            out["FIRE_DETECTED"]["true_fire"] += int(lbl == 1)
        elif p > no_fire_thr:
            out["UNCERTAIN"]["count"] += 1
            out["UNCERTAIN"]["true_fire"] += int(lbl == 1)
        else:
            out["NO_FIRE"]["count"] += 1
            out["NO_FIRE"]["true_fire"] += int(lbl == 1)
    return out


def _plot(float_probs, int8_probs, labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, probs, name in ((axes[0], float_probs, "float32"),
                            (axes[1], int8_probs, "int8")):
        keep = np.where(probs > 1e-6, probs, 1e-6)
        ax.hist(probs[labels == 1], bins=50, alpha=0.6,
                color="#DD8452", label="FIRE", range=(0, 1))
        ax.hist(probs[labels == 0], bins=50, alpha=0.6,
                color="#4C72B0", label="NO_FIRE", range=(0, 1))
        ax.set_title(f"{name} fire-probability (test)"); ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())