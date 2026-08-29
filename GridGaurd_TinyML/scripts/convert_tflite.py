import sys
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI - KERAS TO FLOAT32 TFLITE
# ============================================================
print("=" * 60)
print(" GRIDGUARD AI - FLOAT32 TFLITE CONVERSION")
print("=" * 60)

MODEL_PATH = config.KERAS_MODEL_FILE
TFLITE_PATH = config.TFLITE_FLOAT32_FILE

# ------------------------------------------------------------
# Load Keras model
# ------------------------------------------------------------
print("\nLoading Keras model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Keras model loaded.")

# ------------------------------------------------------------
# Convert to TFLite
# ------------------------------------------------------------
print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
TFLITE_PATH.write_bytes(tflite_model)

size_bytes = TFLITE_PATH.stat().st_size

print("\nFloat32 TFLite model saved:")
print(TFLITE_PATH)
print(f"\nModel size: {size_bytes:,} bytes")
print(f"Model size: {size_bytes / 1024:.2f} KB")

print("\n" + "=" * 60)
print(" FLOAT32 TFLITE CONVERSION COMPLETE")
print("=" * 60)