import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ============================================================
# GRIDGUARD AI
# MODEL 1 - TEMPERATURE ANOMALY DETECTION
# MLP CLASSIFIER
# ============================================================

WINDOW_SIZE = config.WINDOW_SIZE
RANDOM_SEED = config.RANDOM_SEED
EPOCHS = 100
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

config.MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD PREPARED DATA
# ============================================================
print()
print("==========================================")
print(" GRIDGUARD AI - MODEL 1 TRAINING")
print("==========================================")

print()
print("Loading prepared data...")

X_train = np.load(config.X_TRAIN_FILE)
X_val = np.load(config.X_VAL_FILE)
X_test = np.load(config.X_TEST_FILE)

y_train = np.load(config.Y_TRAIN_FILE)
y_val = np.load(config.Y_VAL_FILE)
y_test = np.load(config.Y_TEST_FILE)

print()
print("Training data:  ", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:   ", X_test.shape)

# ============================================================
# BUILD MODEL
# ============================================================
print()
print("Building neural network...")

model = Sequential([
    Input(shape=(WINDOW_SIZE,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])

# ============================================================
# COMPILE MODEL
# ============================================================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ============================================================
# MODEL SUMMARY
# ============================================================
print()
print("Model architecture:")
model.summary()

# ============================================================
# EARLY STOPPING
# ============================================================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

# ============================================================
# TRAIN MODEL
# ============================================================
print()
print("Starting training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1,
)

# ============================================================
# TEST EVALUATION
# ============================================================
print()
print("==========================================")
print(" TEST SET EVALUATION")
print("==========================================")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print()
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# ============================================================
# PREDICTIONS
# ============================================================
print()
print("Generating predictions...")

predictions = model.predict(X_test, verbose=0)
predicted_labels = (predictions >= 0.5).astype(int).flatten()

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, predicted_labels)

print()
print("Confusion Matrix:")
print(cm)
print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        predicted_labels,
        target_names=["Normal", "Anomaly"],
    )
)

# ============================================================
# SAVE MODEL
# ============================================================
MODEL_FILE = config.KERAS_MODEL_FILE
model.save(MODEL_FILE)

print()
print("Model saved to:")
print(MODEL_FILE)

# ============================================================
# TRAINING CURVES
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("GridGuard AI - Model 1 Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ============================================================
# LOSS CURVE
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GridGuard AI - Model 1 Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ============================================================
# COMPLETE
# ============================================================
print()
print("==========================================")
print(" MODEL 1 TRAINING COMPLETE")
print("==========================================")
print()
print(f"Final test accuracy: {test_accuracy * 100:.2f}%")
print()
print("Next step:")
print("Convert the trained model to TensorFlow Lite for TinyML deployment.")
print()
print("==========================================")