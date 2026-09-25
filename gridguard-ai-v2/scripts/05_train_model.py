"""Step 5: GridGuard AI v2 - final model training.

Trains the best architecture found in Step 4 (F_Dense_48_24_12) with the
full training regimen: Adam, sparse categorical crossentropy, early
stopping, model checkpointing and optional LR reduction on plateaus.
Saves training curves (accuracy + loss) and full history.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ACC_LOSS_PNG,
    BATCH_SIZE,
    BEST_CHECKPOINT,
    EARLY_STOP_PATIENCE,
    FINAL_ARCHITECTURE,
    FINAL_MODEL_KERAS,
    HISTORY_CSV,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_FEATURES,
    RANDOM_SEED,
    TEST_NPZ,
    TRAIN_EPOCHS,
    TRAIN_NPZ,
    VAL_NPZ,
)


def set_seed(seed=RANDOM_SEED):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def build_model(layers):
    inputs = tf.keras.Input(shape=(NUM_FEATURES,), name="input")
    x = inputs
    for units in layers:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 5: Final model training")
    print(f"Architecture: {FINAL_ARCHITECTURE} (input=40 -> 48 -> 24 -> 12 -> 5 softmax)")
    print("=" * 72)

    d_train = np.load(TRAIN_NPZ)
    d_val = np.load(VAL_NPZ)
    d_test = np.load(TEST_NPZ)
    X_train, y_train = d_train["X"], d_train["y"]
    X_val, y_val = d_val["X"], d_val["y"]
    X_test, y_test = d_test["X"], d_test["y"]

    set_seed()
    model = build_model(list(FINAL_ARCHITECTURE))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=EARLY_STOP_PATIENCE,
            mode="max", restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_CHECKPOINT),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0
        ),
    ]

    hist = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=TRAIN_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    final_model = tf.keras.models.load_model(BEST_CHECKPOINT)
    final_model.save(FINAL_MODEL_KERAS)
    train_loss, train_acc = final_model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = final_model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)

    print("\nFinal results (best validation checkpoint):")
    print(f"  train acc = {train_acc:.4f}")
    print(f"  val   acc = {val_acc:.4f}")
    print(f"  test  acc = {test_acc:.4f}")
    print(f"  params    = {final_model.count_params()}")
    print(f"Saved Keras model: {FINAL_MODEL_KERAS}")

    with open(HISTORY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(hist.history.keys()))
        writer.writerows(zip(*hist.history.values()))
    print(f"Training history: {HISTORY_CSV}")

    epochs = np.arange(1, len(hist.history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, hist.history["accuracy"], label="train")
    ax1.plot(epochs, hist.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(epochs, hist.history["loss"], label="train")
    ax2.plot(epochs, hist.history["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.suptitle("GridGuard AI v2 - training accuracy & loss")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ACC_LOSS_PNG, dpi=130)
    plt.close(fig)
    print(f"Training curves: {ACC_LOSS_PNG}")


if __name__ == "__main__":
    main()