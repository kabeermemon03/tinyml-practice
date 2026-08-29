import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI - FULL INT8 QUANTIZATION
# ============================================================
print("=" * 60)
print(" GRIDGUARD AI - INT8 QUANTIZATION")
print("=" * 60)

MODEL_PATH = config.KERAS_MODEL_FILE
OUTPUT_PATH = config.TFLITE_INT8_FILE


# ------------------------------------------------------------
# Representative dataset for calibration
# ------------------------------------------------------------
def representative_dataset():
    # 500 samples is more than enough for this small model.
    num_samples = min(500, len(X_train))

    for i in range(num_samples):
        sample = X_train[i:i + 1].astype(np.float32)
        yield [sample]


# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------
print("\nLoading representative dataset...")

X_train = np.load(config.X_TRAIN_FILE)

print(f"Training data shape: {X_train.shape}")

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
print("\nLoading Keras model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Keras model loaded.")

# ------------------------------------------------------------
# Configure converter
# ------------------------------------------------------------
print("\nConfiguring INT8 converter...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Force all operations to INT8.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Input and output are also INT8.
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# ------------------------------------------------------------
# Convert
# ------------------------------------------------------------
print("\nStarting full INT8 quantization...")

tflite_int8_model = converter.convert()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
OUTPUT_PATH.write_bytes(tflite_int8_model)

size_bytes = OUTPUT_PATH.stat().st_size

print("\n" + "=" * 60)
print(" INT8 MODEL CREATED")
print("=" * 60)
print(f"\nSaved to:\n{OUTPUT_PATH}")
print(f"\nModel size:\n{size_bytes:,} bytes")
print(f"\nModel size:\n{size_bytes / 1024:.2f} KB")

print("\n" + "=" * 60)
print(" INT8 QUANTIZATION COMPLETE")
print("=" * 60)