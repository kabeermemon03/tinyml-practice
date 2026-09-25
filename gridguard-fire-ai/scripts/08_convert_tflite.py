"""GridGuard Fire AI - Step 8: Float32 TFLite conversion.

Converts the final Keras model to a float32 TensorFlow Lite model, evaluates
it on the (still untouched) test set with the exact training preprocessing,
and checks numerical parity with the Keras model.

Outputs:
  models/tflite/gridguard_fire_model_float32.tflite
  deployment/gridguard_fire_model.tflite
  models/final/float32_tflite_eval.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf  # noqa: E402

from config import (  # noqa: E402
    CACHE_ROOT,
    DEPLOY_FLOAT_TFLITE,
    FINAL_CONFIG_JSON,
    FINAL_DIR,
    FINAL_MODEL_KERAS,
    FLOAT_TFLITE,
    SPLITS_DIR,
)
from _helpers import build_tf_dataset  # noqa: E402


def main():
    print("=== GridGuard Fire AI | Step 8: float32 TFLite conversion ===")
    cfg = json.loads(FINAL_CONFIG_JSON.read_text(encoding="utf-8"))
    size = cfg["input_size"]

    model = tf.keras.models.load_model(FINAL_MODEL_KERAS)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = converter.convert()
    FLOAT_TFLITE.write_bytes(tflite_bytes)
    DEPLOY_FLOAT_TFLITE.write_bytes(tflite_bytes)
    print(f"Float32 TFLite size : {len(tflite_bytes)} bytes")

    # evaluate on test with the same tf.data preprocessing used in training
    test_cache = CACHE_ROOT / f"cache_{size}x{size}_test"
    ds = build_tf_dataset(SPLITS_DIR / "test.manifest.csv", (size, size),
                          32, train=False, cache_dir=test_cache)

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    tflite_probs, tflite_labels = [], []
    for xb, yb in ds:
        for xi, yi in zip(xb.numpy(), yb.numpy()):
            interp.set_tensor(in_d["index"], xi[None, ...])
            interp.invoke()
            tflite_probs.append(float(interp.get_tensor(out_d["index"])[0, 0]))
            tflite_labels.append(int(yi))
    tflite_probs = np.asarray(tflite_probs)
    tflite_labels = np.asarray(tflite_labels)

    keras_probs = np.load(FINAL_DIR / "test_probs.npy")
    max_diff = float(np.abs(keras_probs - tflite_probs).max())
    mean_diff = float(np.abs(keras_probs - tflite_probs).mean())
    print(f"Keras vs float TFLite parity: max_abs_diff={max_diff:.6f} "
          f"mean_abs_diff={mean_diff:.6f}")

    with open(FINAL_DIR / "float32_tflite_eval.json", "w", encoding="utf-8") as fh:
        json.dump({
            "size_bytes": len(tflite_bytes),
            "max_abs_diff_vs_keras": max_diff,
            "mean_abs_diff_vs_keras": mean_diff,
            "test_images": len(tflite_probs),
            "input_details": {"dtype": str(in_d["dtype"]),
                              "shape": in_d["shape"].tolist(),
                              "scale": in_d.get("quantization", (None, None))[0],
                              "zero_point": in_d.get("quantization", (None, None))[1]},
        }, fh, indent=2)

    with open(FINAL_DIR / "float32_tflite_test_probs.npy", "wb") as fh:
        np.save(fh, tflite_probs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())