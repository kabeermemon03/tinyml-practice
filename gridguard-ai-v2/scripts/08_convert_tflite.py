"""Step 8: GridGuard AI v2 - float32 TFLite conversion + validation.

Converts the trained Keras model to a float32 TensorFlow Lite model and
validates the converted model on the test split against the Keras baseline.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEST_CHECKPOINT, FLOAT_TFLITE, TEST_NPZ


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 8: Float32 TFLite conversion + validation")
    print("=" * 72)
    model = tf.keras.models.load_model(BEST_CHECKPOINT)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = []
    tflite_model = converter.convert()
    FLOAT_TFLITE.write_bytes(tflite_model)
    print(f"Saved float32 TFLite model: {FLOAT_TFLITE} "
          f"({FLOAT_TFLITE.stat().st_size} bytes)")

    d = np.load(TEST_NPZ)
    X_test, y_test = d["X"], d["y"]

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_d, out_d = interpreter.get_input_details(), interpreter.get_output_details()
    print(f"Input tensor : shape={in_d[0]['shape']} dtype={in_d[0]['dtype']}")
    print(f"Output tensor: shape={out_d[0]['shape']} dtype={out_d[0]['dtype']}")

    tflite_probs = []
    for x in X_test:
        interpreter.set_tensor(in_d[0]["index"], x.reshape(-1, 40).astype(np.float32))
        interpreter.invoke()
        tflite_probs.append(interpreter.get_tensor(out_d[0]["index"]).copy())
    tflite_probs = np.array(tflite_probs).reshape(len(X_test), -1)
    tflite_preds = tflite_probs.argmax(axis=1)

    keras_probs = model.predict(X_test, verbose=0)
    keras_acc = float((keras_probs.argmax(1) == y_test).mean())
    tflite_acc = float((tflite_preds == y_test).mean())
    max_diff = float(np.abs(keras_probs - tflite_probs).max())

    print(f"\nKeras  test accuracy : {keras_acc:.4f}")
    print(f"Float TFLite accuracy: {tflite_acc:.4f}")
    print(f"Max |keras - tflite| probability diff: {max_diff:.2e}")
    assert tflite_acc >= keras_acc - 0.001, "unexpected float conversion drift"
    print("Float conversion validated OK.")


if __name__ == "__main__":
    main()