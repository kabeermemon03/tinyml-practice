import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI - FLOAT32 TFLITE VALIDATION
# ============================================================
print("=" * 60)
print(" GRIDGUARD AI - FLOAT32 TFLITE VALIDATION")
print("=" * 60)

MODEL_PATH = config.KERAS_MODEL_FILE
TFLITE_PATH = config.TFLITE_FLOAT32_FILE

# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------
print("\nLoading test data...")

X_test = np.load(config.X_TEST_FILE)
y_test = np.load(config.Y_TEST_FILE)

print(f"Test samples: {len(X_test)}")
print(f"Input shape:  {X_test.shape}")

# ------------------------------------------------------------
# Keras predictions
# ------------------------------------------------------------
print("\nLoading Keras model...")

keras_model = tf.keras.models.load_model(MODEL_PATH)

print("Generating Keras predictions...")

keras_probs = keras_model.predict(X_test, verbose=0).flatten()

# ------------------------------------------------------------
# Load TFLite model
# ------------------------------------------------------------
print("\nLoading Float32 TFLite model...")

interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nTFLite input:")
print(input_details)
print("\nTFLite output:")
print(output_details)

# ------------------------------------------------------------
# TFLite predictions
# ------------------------------------------------------------
print("\nGenerating TFLite predictions...")

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

tflite_probs = []

for i in range(len(X_test)):
    input_data = X_test[i:i + 1].astype(np.float32)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)
    tflite_probs.append(output[0][0])

tflite_probs = np.array(tflite_probs)

# ------------------------------------------------------------
# Compare probabilities
# ------------------------------------------------------------
print("\n" + "=" * 60)
print(" PROBABILITY COMPARISON")
print("=" * 60)

difference = np.abs(keras_probs - tflite_probs)

print(f"\nMaximum difference : {difference.max():.10f}")
print(f"Mean difference    : {difference.mean():.10f}")
print(f"Median difference  : {np.median(difference):.10f}")

# ------------------------------------------------------------
# Accuracy comparison
# ------------------------------------------------------------
threshold = config.DEFAULT_THRESHOLD

keras_pred = (keras_probs >= threshold).astype(int)
tflite_pred = (tflite_probs >= threshold).astype(int)

keras_accuracy = accuracy_score(y_test, keras_pred)
tflite_accuracy = accuracy_score(y_test, tflite_pred)

print("\n" + "=" * 60)
print(" ACCURACY COMPARISON")
print("=" * 60)
print(f"\nThreshold: {threshold}")
print(f"Keras accuracy : {keras_accuracy * 100:.2f}%")
print(f"TFLite accuracy: {tflite_accuracy * 100:.2f}%")

# ------------------------------------------------------------
# Confusion matrices
# ------------------------------------------------------------
print("\nKeras confusion matrix:")
print(confusion_matrix(y_test, keras_pred))
print("\nTFLite confusion matrix:")
print(confusion_matrix(y_test, tflite_pred))

# ------------------------------------------------------------
# Prediction agreement
# ------------------------------------------------------------
agreement = np.mean(keras_pred == tflite_pred)

print("\n" + "=" * 60)
print(" PREDICTION AGREEMENT")
print("=" * 60)
print(f"\nClassification agreement: {agreement * 100:.2f}%")

# ------------------------------------------------------------
# Different classifications
# ------------------------------------------------------------
different = np.where(keras_pred != tflite_pred)[0]

print(f"Different classifications: {len(different)}")

if len(different) > 0:
    print("\nDifferences:")
    for i in different[:20]:
        print(
            f"Index {i}: Keras={keras_probs[i]:.6f}, "
            f"TFLite={tflite_probs[i]:.6f}, Actual={y_test[i]}"
        )

# ------------------------------------------------------------
# Sample comparison
# ------------------------------------------------------------
print("\n" + "=" * 60)
print(" SAMPLE PREDICTION COMPARISON")
print("=" * 60)

for i in range(10):
    print(f"\nSample {i + 1}:")
    print(f"  Keras :   {keras_probs[i]:.8f}")
    print(f"  TFLite:   {tflite_probs[i]:.8f}")
    print(f"  Difference: {difference[i]:.10f}")

# ------------------------------------------------------------
# Final verdict
# ------------------------------------------------------------
print("\n" + "=" * 60)
print(" FLOAT32 TFLITE VALIDATION COMPLETE")
print("=" * 60)

if len(different) == 0:
    print("\nPASS: Keras and TFLite classifications match.")
else:
    print("\nWARNING: Some classifications differ.")

print("\nNext step: INT8 quantization")