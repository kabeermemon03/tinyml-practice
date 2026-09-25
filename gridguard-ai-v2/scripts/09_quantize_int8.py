"""Step 9: GridGuard AI v2 - full INT8 quantization + evaluation.

Pipeline:
  1. representative dataset drawn from TRAINING split only (no leakage)
  2. full integer quantization for TensorFlow Lite Micro (int8 I/O)
  3. evaluate INT8 model on the test split and compare with the float
     models
  4. print model sizes and all quantization parameters
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FLOAT_TFLITE,
    FLOAT_VS_INT8_PNG,
    INT8_TFLITE,
    NUM_FEATURES,
    QUANT_PARAMS_JSON,
    TEST_NPZ,
    TRAIN_NPZ,
)

REP_DATASET_SAMPLES = 2000


def representative_dataset():
    d = np.load(TRAIN_NPZ)
    X = d["X"][:REP_DATASET_SAMPLES]
    for x in X:
        yield [x.reshape(1, NUM_FEATURES).astype(np.float32)]


def evaluate_interpreter(interpreter, X, y):
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    in_scale, in_zp = in_d["quantization"]
    out_scale, out_zp = out_d["quantization"]
    preds = []
    for x in X:
        q = np.clip(np.round(x / in_scale) + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(in_d["index"], q.reshape(1, -1))
        interpreter.invoke()
        raw = interpreter.get_tensor(out_d["index"]).reshape(-1)
        probs = (raw.astype(np.float32) - out_zp) * out_scale
        preds.append(int(np.argmax(probs)))
    return float(np.mean(np.array(preds) == y))


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 9: INT8 quantization + eval")
    print("=" * 72)

    d = np.load(TRAIN_NPZ)
    X_train, y_train = d["X"], d["y"]
    d2 = np.load(TEST_NPZ)
    X_test, y_test = d2["X"], d2["y"]
    print(f"Rep dataset source: {REP_DATASET_SAMPLES} training examples "
          f"(train split only, no leakage)")

    keras_model_path = Path(__file__).resolve().parent.parent / "models" / "gridguard_v2_best.keras"
    model = tf.keras.models.load_model(keras_model_path)
    keras_size = (Path(__file__).resolve().parent.parent / "models" / "gridguard_v2_final.keras").stat().st_size
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_model = converter.convert()
    INT8_TFLITE.write_bytes(int8_model)
    int8_size = INT8_TFLITE.stat().st_size
    float_size = FLOAT_TFLITE.stat().st_size
    print(f"Saved INT8 TFLite model: {INT8_TFLITE} ({int8_size} bytes)")
    print(f"Float32 TFLite          : {float_size} bytes")
    print(f"Keras (.keras)          : {keras_size} bytes")

    interp = tf.lite.Interpreter(model_content=int8_model)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"INT8 input  : shape={in_d['shape']} dtype={in_d['dtype']} "
          f"scale={in_d['quantization'][0]:.6f} zero_point={in_d['quantization'][1]}")
    print(f"INT8 output : shape={out_d['shape']} dtype={out_d['dtype']} "
          f"scale={out_d['quantization'][0]:.6f} zero_point={out_d['quantization'][1]}")

    keras_model = tf.keras.models.load_model(keras_model_path)
    float_acc = float((keras_model.predict(X_test, verbose=0).argmax(1) == y_test).mean())

    float_interp = tf.lite.Interpreter(model_path=str(FLOAT_TFLITE))
    float_interp.allocate_tensors()
    f_in = float_interp.get_input_details()[0]
    float_preds = []
    for x in X_test:
        float_interp.set_tensor(f_in["index"], x.reshape(1, -1).astype(np.float32))
        float_interp.invoke()
        float_preds.append(int(float_interp.get_tensor(
            float_interp.get_output_details()[0]["index"]).argmax()))
    float_tflite_acc = float(np.mean(np.array(float_preds) == y_test))

    int8_acc = evaluate_interpreter(interp, X_test, y_test)
    print(f"\nKeras test accuracy   : {float_acc:.4f}")
    print(f"Float32 TFLite acc    : {float_tflite_acc:.4f}")
    print(f"INT8 TFLite accuracy  : {int8_acc:.4f}")
    print(f"Accuracy drop (float->int8): {(float_tflite_acc - int8_acc):.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Keras\n(float)", "TFLite\n(float32)", "TFLite\n(INT8)"]
    accs = [float_acc, float_tflite_acc, int8_acc]
    sizes = [float_size, float_size, int8_size]
    colors = ["#1565c0", "#42a5f5", "#c62828"]
    bars = ax.bar(labels, accs, color=colors, alpha=0.85)
    for bar, a, s in zip(bars, accs, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.004,
                f"{a:.4f}\n({s} B)", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy comparison: float vs INT8")
    fig.tight_layout()
    fig.savefig(FLOAT_VS_INT8_PNG, dpi=130)
    plt.close(fig)
    print(f"Comparison chart: {FLOAT_VS_INT8_PNG}")

    all_tensors = []
    for det in interp.get_tensor_details():
        scale, zp = det["quantization"]
        all_tensors.append({
            "name": det["name"],
            "shape": [int(dim) for dim in det["shape"]],
            "dtype": str(det["dtype"]),
            "scale": float(scale) if scale else None,
            "zero_point": int(zp) if zp is not None else None,
        })
    params = {
        "model_file": INT8_TFLITE.relative_to(INT8_TFLITE.parents[1]).as_posix(),
        "model_size_bytes": int8_size,
        "float_model_size_bytes": float_size,
        "input": {
            "shape": [int(s) for s in in_d["shape"]],
            "dtype": str(in_d["dtype"]),
            "scale": float(in_d["quantization"][0]),
            "zero_point": int(in_d["quantization"][1]),
            "name": in_d["name"],
        },
        "output": {
            "shape": [int(s) for s in out_d["shape"]],
            "dtype": str(out_d["dtype"]),
            "scale": float(out_d["quantization"][0]),
            "zero_point": int(out_d["quantization"][1]),
            "name": out_d["name"],
        },
        "all_tensors": all_tensors,
    }
    with open(QUANT_PARAMS_JSON, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Quantization parameters: {QUANT_PARAMS_JSON}")


if __name__ == "__main__":
    main()