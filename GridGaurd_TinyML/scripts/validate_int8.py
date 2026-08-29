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
# GRIDGUARD AI - INT8 MODEL VALIDATION
# ============================================================

KERAS_MODEL = config.KERAS_MODEL_FILE
INT8_MODEL = config.TFLITE_INT8_FILE

WINDOW_SIZE = config.WINDOW_SIZE

# Experimental detection threshold used during validation comparisons.
THRESHOLD = 0.1


print("=" * 60)
print(" GRIDGUARD AI - INT8 MODEL VALIDATION")
print("=" * 60)

# ============================================================
# LOAD TEST DATA
# ============================================================
print("\nLoading test data...")

X_test = np.load(config.X_TEST_FILE)
y_test = np.load(config.Y_TEST_FILE)

print(f"Test samples : {len(X_test)}")
print(f"Input shape  : {X_test.shape}")
print(f"Labels shape : {y_test.shape}")

# ============================================================
# LOAD KERAS MODEL (baseline)
# ============================================================
print("\nLoading Keras baseline model...")

keras_model = tf.keras.models.load_model(KERAS_MODEL)

print("Keras model loaded.")

# ============================================================
# KERAS PREDICTIONS
# ============================================================
print("\nGenerating Keras predictions...")

keras_probs = keras_model.predict(X_test, verbose=0).flatten()
keras_preds = (keras_probs >= THRESHOLD).astype(int)

keras_accuracy = accuracy_score(y_test, keras_preds)
keras_cm = confusion_matrix(y_test, keras_preds)

# ============================================================
# LOAD INT8 TFLITE MODEL
# ============================================================
print("\nLoading INT8 TFLite model...")

interpreter = tf.lite.Interpreter(model_path=str(INT8_MODEL))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INT8 model loaded.")

# ============================================================
# MODEL INFORMATION
# ============================================================
input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

model_size = INT8_MODEL.stat().st_size

print("\n" + "=" * 60)
print(" INT8 MODEL INFORMATION")
print("=" * 60)
print(f"\nModel size: {model_size:,} bytes")
print(f"Model size: {model_size / 1024:.2f} KB")
print(f"\nInput shape : {input_details[0]['shape']}")
print(f"Input dtype : {input_details[0]['dtype']}")
print(f"Input quant : scale={input_scale}, zero_point={input_zero_point}")
print(f"\nOutput dtype: {output_details[0]['dtype']}")
print(f"Output quant: scale={output_scale}, zero_point={output_zero_point}")

# ============================================================
# VERIFY INT8 TYPES
# ============================================================
input_is_int8 = input_details[0]["dtype"] == np.int8
output_is_int8 = output_details[0]["dtype"] == np.int8

print("\n" + "=" * 60)
print(" QUANTIZATION TYPE CHECK")
print("=" * 60)
print(f"\nInput INT8  : {'PASS' if input_is_int8 else 'FAIL'}")
print(f"Output INT8 : {'PASS' if output_is_int8 else 'FAIL'}")

if not input_is_int8 or not output_is_int8:
    print("\nWARNING: Model is not fully INT8.")
    print("Validation stopped.")
    raise SystemExit(1)

if input_scale <= 0 or output_scale <= 0:
    print("\nERROR: Invalid quantization scale.")
    raise SystemExit(1)

# ============================================================
# INT8 PREDICTIONS
#
# Quantization:  q = round(x / scale + zero_point)
# Dequantization: x = (q - zero_point) * scale
# ============================================================
print("\n" + "=" * 60)
print(" GENERATING INT8 PREDICTIONS")
print("=" * 60)

int8_probs = []

for i in range(len(X_test)):
    # FLOAT32 -> INT8 (clip to valid INT8 range)
    x_quantized = np.round(
        X_test[i] / np.float32(input_scale) + np.float32(input_zero_point)
    )
    x_quantized = np.clip(x_quantized, -128, 127).astype(np.int8)
    x_quantized = x_quantized.reshape(1, WINDOW_SIZE)

    interpreter.set_tensor(input_details[0]["index"], x_quantized)
    interpreter.invoke()

    output_quantized = interpreter.get_tensor(output_details[0]["index"])[0][0]

    # CRITICAL: convert INT8 -> INT32 BEFORE subtracting the zero point,
    # otherwise NumPy can overflow during INT8 arithmetic.
    probability = (
        np.int32(output_quantized) - np.int32(output_zero_point)
    ) * np.float32(output_scale)

    # Safety clamp: sigmoid output should be in [0, 1].
    int8_probs.append(float(np.clip(probability, 0.0, 1.0)))

int8_probs = np.array(int8_probs, dtype=np.float32)

# ============================================================
# INT8 CLASSIFICATION
# ============================================================
int8_preds = (int8_probs >= THRESHOLD).astype(int)

int8_accuracy = accuracy_score(y_test, int8_preds)
int8_cm = confusion_matrix(y_test, int8_preds)

print("\n" + "=" * 60)
print(" ACCURACY COMPARISON")
print("=" * 60)
print(f"\nThreshold: {THRESHOLD}")
print(f"Keras accuracy       : {keras_accuracy * 100:.2f}%")
print(f"INT8 TFLite accuracy : {int8_accuracy * 100:.2f}%")

print("\nKeras confusion matrix:")
print(keras_cm)
print("\nINT8 confusion matrix:")
print(int8_cm)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 60)
print(" INT8 CLASSIFICATION REPORT")
print("=" * 60)
print(
    classification_report(
        y_test,
        int8_preds,
        target_names=["Normal", "Anomaly"],
        digits=4,
    )
)

# ============================================================
# PROBABILITY COMPARISON
# ============================================================
print("\n" + "=" * 60)
print(" PROBABILITY COMPARISON")
print("=" * 60)

prob_difference = np.abs(keras_probs - int8_probs)

print(f"\nMaximum difference : {np.max(prob_difference):.8f}")
print(f"Mean difference    : {np.mean(prob_difference):.8f}")
print(f"Median difference  : {np.median(prob_difference):.8f}")

# ============================================================
# PREDICTION AGREEMENT
# ============================================================
agreement = np.mean(keras_preds == int8_preds)
different = np.sum(keras_preds != int8_preds)

print("\n" + "=" * 60)
print(" PREDICTION AGREEMENT")
print("=" * 60)
print(f"\nClassification agreement: {agreement * 100:.2f}%")
print(f"Different classifications: {different}")

# ============================================================
# MISCLASSIFIED INT8 SAMPLES
# ============================================================
misclassified = np.where(int8_preds != y_test)[0]

print("\n" + "=" * 60)
print(" INT8 MISCLASSIFIED SAMPLES")
print("=" * 60)
print(f"\nTotal misclassified: {len(misclassified)}")

for index in misclassified:
    actual_label = "Anomaly" if y_test[index] == 1 else "Normal"
    predicted_label = "Anomaly" if int8_preds[index] == 1 else "Normal"

    print(
        f"Index {index:3d} | "
        f"Actual: {actual_label:7s} | "
        f"Predicted: {predicted_label:7s} | "
        f"Probability: {int8_probs[index]:.6f}"
    )

# ============================================================
# SAMPLE KERAS vs INT8 PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print(" SAMPLE KERAS vs INT8 PREDICTIONS")
print("=" * 60)

for i in range(min(10, len(X_test))):
    print(f"\nSample {i + 1}:")
    print(f"  Actual : {'Anomaly' if y_test[i] == 1 else 'Normal'}")
    print(f"  Keras  : {keras_probs[i]:.8f}")
    print(f"  INT8   : {int8_probs[i]:.8f}")
    print(f"  Diff   : {prob_difference[i]:.8f}")

# ============================================================
# RAW INT8 OUTPUT INSPECTION
# ============================================================
print("\n" + "=" * 60)
print(" RAW INT8 OUTPUT INSPECTION")
print("=" * 60)
print("\nChecking first 10 samples...")

for i in range(min(10, len(X_test))):
    x_quantized = np.round(
        X_test[i] / np.float32(input_scale) + np.float32(input_zero_point)
    )
    x_quantized = np.clip(x_quantized, -128, 127).astype(np.int8)

    interpreter.set_tensor(
        input_details[0]["index"], x_quantized.reshape(1, WINDOW_SIZE)
    )
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])[0][0]
    dequantized = (
        np.int32(raw_output) - np.int32(output_zero_point)
    ) * np.float32(output_scale)

    print(f"\nSample {i + 1}:")
    print(f"  Raw INT8 output : {int(raw_output)}")
    print(f"  Dequantized     : {float(dequantized):.8f}")

# ============================================================
# FINAL VERDICT
# ============================================================
print("\n" + "=" * 60)
print(" INT8 VALIDATION RESULT")
print("=" * 60)

accuracy_difference = (keras_accuracy - int8_accuracy) * 100

print(f"\nAccuracy difference: {accuracy_difference:+.2f} percentage points")

if int8_accuracy >= 0.98 and agreement >= 0.99 and different == 0:
    print("\nPASS: INT8 model is suitable for TinyML deployment.")
elif int8_accuracy >= 0.95:
    print("\nPASS WITH REVIEW: INT8 model performs well, but prediction differences exist.")
else:
    print("\nREVIEW REQUIRED: INT8 quantization caused significant accuracy degradation.")

print("\n" + "=" * 60)
print(" INT8 VALIDATION COMPLETE")
print("=" * 60)