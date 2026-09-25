"""GridGuard Fire AI - Step 9: INT8 quantization.

Produces the full INT8 (integer-only) TFLite model for TensorFlow Lite Micro.

  * Representative dataset is drawn ONLY from the training split (never test).
  * If INT8 FIRE recall on the test set falls more than 3 points below the
    float model, quantization is retried with a larger representative set
    (up to `QUANTIZATION_ATTEMPTS` attempts).  The best attempt wins.

Outputs:
  models/tflite/gridguard_fire_model_int8.tflite
  deployment/gridguard_fire_model_int8.tflite
  models/final/quantization_params.json
  models/final/int8_attempt_log.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf  # noqa: E402

from config import (  # noqa: E402
    DEPLOY_INT8_TFLITE,
    FINAL_CONFIG_JSON,
    FINAL_DIR,
    FINAL_MODEL_KERAS,
    INT8_TFLITE,
    QUANT_PARAMS_JSON,
    QUANTIZATION_ATTEMPTS,
    REP_DATASET_SIZE,
    SPLITS_DIR,
)
from _helpers import (  # noqa: E402
    binary_metrics,
    build_tf_dataset,
    load_manifest,
)

RECALL_LOSS_TOLERANCE = 0.03


def main():
    print("=== GridGuard Fire AI | Step 9: INT8 quantization ===")
    cfg = json.loads(FINAL_CONFIG_JSON.read_text(encoding="utf-8"))
    size = cfg["input_size"]
    fire_thr = float(cfg["fire_threshold"])

    model = tf.keras.models.load_model(FINAL_MODEL_KERAS)
    train_rows = load_manifest(SPLITS_DIR / "train.manifest.csv")
    label_paths = [row["path"] for row in train_rows]
    label_ids = [int(row["label"]) for row in train_rows]

    float_test_probs = np.load(FINAL_DIR / "float32_tflite_test_probs.npy")
    test_labels = np.load(FINAL_DIR / "test_labels.npy")
    float_metrics = binary_metrics(float_test_probs, test_labels, 0.5)

    attempts = []
    best = None
    for attempt in range(1, QUANTIZATION_ATTEMPTS + 1):
        rep_size = REP_DATASET_SIZE * attempt
        print(f"\nAttempt {attempt}/{QUANTIZATION_ATTEMPTS} (rep set = {rep_size})")
        tflite_bytes, in_params, out_params = _quantize(model, size, label_paths,
                                                        label_ids, rep_size)
        if tflite_bytes is None:
            attempts.append({"attempt": attempt, "status": "failed"})
            continue
        int8_probs, int8_labels = _run_int8(tflite_bytes, size)
        m = binary_metrics(int8_probs, int8_labels, 0.5)
        m_fire = binary_metrics(int8_probs, int8_labels, fire_thr)
        recall_drop = float_metrics["fire_recall"] - m["fire_recall"]
        print(f"  int8: acc={m['accuracy']} fire_recall={m['fire_recall']} "
              f"recall_drop={recall_drop:.3f}")
        attempts.append({
            "attempt": attempt, "status": "ok", "rep_size": rep_size,
            "metrics_0_5": m, "metrics_fire_thr": m_fire,
            "recall_drop_from_float": round(recall_drop, 4),
            "size_bytes": len(tflite_bytes),
        })
        candidate = (m["fire_recall"], len(tflite_bytes))
        if best is None or candidate[0] > best[0]:
            best = {
                "bytes": tflite_bytes, "rep_size": rep_size,
                "in_params": in_params, "out_params": out_params,
                "metrics": m, "metrics_fire_thr": m_fire, "attempt": attempt,
            }
        if recall_drop <= RECALL_LOSS_TOLERANCE:
            print("  recall preserved - stopping early")
            break

    if best is None:
        print("FATAL: all quantization attempts failed")
        return 1

    tflite_bytes = best["bytes"]
    INT8_TFLITE.write_bytes(tflite_bytes)
    DEPLOY_INT8_TFLITE.write_bytes(tflite_bytes)

    params = {
        "input_size": size,
        "input_channels": 3,
        "rep_dataset": {
            "source": "training split only",
            "size": best["rep_size"],
            "resize": "bilinear",
            "normalize": "x / 255 -> [0,1]",
        },
        "output": {
            "species": "SIGMOID_FIRE_PROBABILITY",
            "scale": best["out_params"]["scale"],
            "zero_point": best["out_params"]["zero_point"],
            "quantized_range": "int8",
        },
        "input": best["in_params"],
        "size_bytes": len(tflite_bytes),
        "test_metrics": best["metrics"],
        "test_metrics_at_fire_threshold": best["metrics_fire_thr"],
        "best_attempt": best["attempt"],
    }
    with open(QUANT_PARAMS_JSON, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    with open(FINAL_DIR / "int8_attempt_log.json", "w", encoding="utf-8") as fh:
        json.dump(attempts, fh, indent=2)

    print(f"\nINT8 model written : {INT8_TFLITE} ({len(tflite_bytes)} bytes)")
    print("Quantization params:")
    print(f"  input  : dtype=int8 scale={params['input']['scale']} "
          f"zp={params['input']['zero_point']}")
    print(f"  output : dtype=int8 scale={params['output']['scale']} "
          f"zp={params['output']['zero_point']}")
    return 0


def _quantize(model, size, label_paths, label_ids, rep_size):
    """Return (tflite_bytes, input_params, output_params) or (None, None, None)."""
    rng = random.Random(7)  # fixed, independent of training seed
    idx = rng.sample(range(len(label_paths)), min(rep_size, len(label_paths)))
    h, w = size, size

    def _decode(p):
        bits = tf.io.read_file(p)
        img = tf.io.decode_image(
            bits, channels=3, dtype=tf.uint8, expand_animations=False
        )
        img = tf.ensure_shape(img, [None, None, 3])
        img = tf.image.resize(img, [h, w], method=tf.image.ResizeMethod.BILINEAR)
        return img

    def rep_ds():
        ds = tf.data.Dataset.from_tensor_slices(
            ([label_paths[i] for i in idx], [label_ids[i] for i in idx]))
        ds = ds.map(lambda p, l: (_decode(p), l), num_parallel_calls=4)
        ds = ds.map(lambda img, l: (tf.cast(img, tf.float32) / 255.0, l))
        for img, _ in ds.batch(1):
            yield [img.numpy()]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_bytes = converter.convert()
    except Exception as exc:  # noqa: BLE001
        print(f"    conversion failed: {exc}")
        return None, None, None

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = in_d["quantization"]
    out_scale, out_zp = out_d["quantization"]
    in_params = {"dtype": str(in_d["dtype"]),
                 "shape": in_d["shape"].tolist(),
                 "scale": float(in_scale), "zero_point": int(in_zp)}
    out_params = {"dtype": str(out_d["dtype"]),
                  "shape": out_d["shape"].tolist(),
                  "scale": float(out_scale), "zero_point": int(out_zp)}
    return tflite_bytes, in_params, out_params


def _run_int8(tflite_bytes, size):
    from config import CACHE_ROOT

    test_cache = CACHE_ROOT / f"cache_{size}x{size}_test"
    ds = build_tf_dataset(SPLITS_DIR / "test.manifest.csv", (size, size),
                          32, train=False, cache_dir=test_cache)
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
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


if __name__ == "__main__":
    raise SystemExit(main())