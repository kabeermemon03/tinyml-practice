"""Step 4: GridGuard AI v2 - architecture experiments.

Trains several small MLPs on the same splits and compares:
  * training/validation accuracy
  * number of parameters
  * float32 TFLite model size (bytes)
Picks the smallest architecture with strong validation accuracy for the
final ESP32-S3 deployable model.
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ARCH_COMPARISON_PNG,
    ARCH_EXPERIMENTS_CSV,
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOP_PATIENCE,
    FLOAT_TFLITE,
    NUM_CLASSES,
    NUM_FEATURES,
    RANDOM_SEED,
    TEST_NPZ,
    TRAIN_EPOCHS,
    TRAIN_NPZ,
    VAL_NPZ,
)

ARCHITECTURES = {
    "A_Dense_32_16_8": (32, 16, 8),
    "B_Dense_64_32_16": (64, 32, 16),
    "C_Dense_24_12": (24, 12),
    "D_Dense_16": (16,),
    "E_Dense_32_16": (32, 16),
    "F_Dense_48_24_12": (48, 24, 12),
}


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
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 4: Architecture experiments")
    print("=" * 72)

    d_train = np.load(TRAIN_NPZ)
    d_val = np.load(VAL_NPZ)
    d_test = np.load(TEST_NPZ)
    X_train, y_train = d_train["X"], d_train["y"]
    X_val, y_val = d_val["X"], d_val["y"]
    X_test, y_test = d_test["X"], d_test["y"]
    print(f"train {X_train.shape} | val {X_val.shape} | test {X_test.shape}")

    results = []
    for name, layers in ARCHITECTURES.items():
        set_seed()
        model = build_model(layers)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
            )
        ]
        t0 = time.time()
        hist = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=TRAIN_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0,
        )
        elapsed = time.time() - t0
        best_epoch = int(np.argmax(hist.history["val_accuracy"])) + 1
        train_acc = float(np.max(hist.history["accuracy"]))
        val_acc = float(np.max(hist.history["val_accuracy"]))
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        n_params = model.count_params()

        tmp_tflite = FLOAT_TFLITE.with_name(f"_tmp_{name}.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tmp_tflite.write_bytes(tflite_model)
        size_bytes = tmp_tflite.stat().st_size
        tmp_tflite.unlink()

        results.append({
            "architecture": name,
            "layers": str(list(layers)),
            "params": int(n_params),
            "best_epoch": best_epoch,
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "test_acc": round(float(test_acc), 4),
            "float_tflite_bytes": int(size_bytes),
            "train_time_sec": round(elapsed, 1),
        })
        print(f"  {name:<20} params={n_params:>6} best_ep={best_epoch:>2} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={float(test_acc):.4f} "
              f"tflite={size_bytes} B  ({elapsed:.1f}s)")
        tf.keras.backend.clear_session()

    with open(ARCH_EXPERIMENTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparison table: {ARCH_EXPERIMENTS_CSV}")

    best = max(results, key=lambda r: (r["val_acc"], -r["params"]))
    print(f"\nBest candidate: {best['architecture']} "
          f"(val_acc={best['val_acc']}, params={best['params']}, tflite={best['float_tflite_bytes']} B)")

    names = [r["architecture"] for r in results]
    val = [r["val_acc"] for r in results]
    sizes = [r["float_tflite_bytes"] for r in results]
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    ax1.bar(x - 0.2, val, 0.4, label="val acc", color="#1565c0")
    ax1.set_ylabel("validation accuracy")
    ax1.set_ylim(0.7, 1.0)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, sizes, 0.4, label="float tflite bytes", color="#c62828")
    ax2.set_ylabel("tflite size (bytes)")
    ax1.set_xticks(x, [n.split("_")[0] for n in names])
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")
    ax1.set_title("Architecture comparison: validation accuracy vs model size")
    fig.tight_layout()
    fig.savefig(ARCH_COMPARISON_PNG, dpi=130)
    plt.close(fig)
    print(f"Comparison chart: {ARCH_COMPARISON_PNG}")


if __name__ == "__main__":
    main()