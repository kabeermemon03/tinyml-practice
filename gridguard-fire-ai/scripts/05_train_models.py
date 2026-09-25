"""GridGuard Fire AI - Step 5: Training experiments.

Runs three experiment groups on the leakage-safe manifests:

  Phase 1 - Resolution sweep:      tiny_cnn trained at 64 / 96 / 128.
  Phase 2 - Architecture sweep:    deep_cnn / separable_cnn / tinymobilenet
                                   trained at the selected resolution.
  Phase 3 - Transfer reference:    frozen MobileNetV2 (imagenet) backbone +
                                   new head - used ONLY as a performance
                                   reference; NOT a deployment candidate.

Every experiment records: parameters, best validation metrics, float32 TFLite
size, INT8 TFLite size + validation accuracy, and validation predictions so
later steps never need to rerun training.

Usage examples:
  python 05_train_models.py                       # run everything
  python 05_train_models.py --skip-transfer       # only scratch models
  python 05_train_models.py --only tiny_cnn_96    # single experiment
  python 05_train_models.py --arch-size 96 --epochs 40 --batch 64
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf

from config import (  # noqa: E402
    ARCHITECTURES,
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    CACHE_ROOT,
    COMPARISON_CSV,
    DO_TRANSFER_LEARNING,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENTS_DIR,
    FINAL_DIR,
    FOCAL_GAMMA,
    LEARNING_RATE,
    PROCESSED_DIR,
    RESOLUTION_CANDIDATES,
    SPLITS_DIR,
    TRANSFER_ARCH,
    TRANSFER_IMAGE_SIZE,
    TRAIN_EPOCHS,
    WEIGHT_DECAY,
)
from _helpers import (  # noqa: E402
    FocalBinaryCrossentropy,
    binary_metrics,
    build_tf_dataset,
    predict_probs,
    set_all_seeds,
)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
def tiny_cnn(input_shape):
    """Model A - minimal 3-block CNN with GAP (fewest parameters)."""
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, x, name="tiny_cnn")


def deep_cnn(input_shape):
    """Model B - slightly deeper 4-block CNN (more capacity)."""
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, x, name="deep_cnn")


def separable_cnn(input_shape):
    """Model C - depthwise-separable 3-block CNN (few params, less compute)."""
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.SeparableConv2D(24, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.SeparableConv2D(48, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.SeparableConv2D(96, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, x, name="separable_cnn")


def tinymobilenet(input_shape):
    """Model D - compact depthwise-separable (MobileNet-Nano style)."""
    def dw(inp, out, stride):
        x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same")(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(out, 1, 1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = dw(x, 24, 1)
    x = dw(x, 32, 2)
    x = dw(x, 48, 1)
    x = dw(x, 64, 1)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, x, name="tinymobilenet")


def mobilenet_v2(input_shape):
    """Transfer reference: frozen MobileNetV2 (imagenet) + new head."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    inp = tf.keras.Input(shape=input_shape)
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, x, name="mobilenet_v2_head")


ARCH_BUILDERS = {
    "tiny_cnn": tiny_cnn,
    "deep_cnn": deep_cnn,
    "separable_cnn": separable_cnn,
    "tinymobilenet": tinymobilenet,
    TRANSFER_ARCH: mobilenet_v2,
}


# ---------------------------------------------------------------------------
# Classification weights
# ---------------------------------------------------------------------------
def class_weights(labels):
    n = len(labels)
    counts = {0: int(np.sum(labels == 0)), 1: int(np.sum(labels == 1))}
    return {0: n / (2 * counts[0]), 1: n / (2 * counts[1])}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(exp_id, arch, image_size, batch_size, epochs, train_manifest,
                   val_manifest, seed, override_epochs=None):
    set_all_seeds(seed)
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n>>> Experiment {exp_id}: arch={arch} size={image_size} "
          f"batch={batch_size} epochs={epochs}")

    h, w = image_size, image_size

    train_cache = CACHE_ROOT / f"cache_{h}x{w}_train"
    val_cache = CACHE_ROOT / f"cache_{h}x{w}_val"

    train_ds = build_tf_dataset(train_manifest, (h, w), batch_size, train=True,
                                cache_dir=train_cache)
    val_ds = build_tf_dataset(val_manifest, (h, w), batch_size, train=False,
                              cache_dir=val_cache)

    model = ARCH_BUILDERS[arch]((h, w, 3))
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        ),
        loss=FocalBinaryCrossentropy(gamma=FOCAL_GAMMA, alpha=0.6),
        metrics=["accuracy"],
    )
    params = model.count_params()

    checkpoints = CHECKPOINTS_DIR / f"{exp_id}"
    checkpoints.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoints / "best.keras", monitor="val_loss",
            save_best_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
    ]

    # labels for class weighting
    _, train_labels = _labels_from_manifest(train_manifest)
    cw = class_weights(np.asarray(train_labels))

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=cw, callbacks=callbacks, verbose=1,
    )

    # best checkpoint metrics on validation (fall back to final weights if a
    # best checkpoint was never written, e.g. no val improvement at all)
    best_path = checkpoints / "best.keras"
    if best_path.exists():
        best = tf.keras.models.load_model(best_path)
    else:
        best = model
    probs, labels = predict_probs(best, val_manifest, (h, w), batch_size,
                                  cache_dir=val_cache)
    np.save(exp_dir / "val_probs.npy", probs)
    np.save(exp_dir / "val_labels.npy", labels)

    m50 = binary_metrics(probs, labels, 0.5)

    # ---- float32 TFLite (size for comparison) ----
    float_tflite = exp_dir / "float32.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(best)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    float_bytes = converter.convert()
    float_tflite.write_bytes(float_bytes)

    # ---- INT8 TFLite (size + quick validation accuracy for comparison) ----
    int8_tflite = exp_dir / "int8.tflite"
    int8_acc, int8_probs = _make_int8(best, (h, w), int8_tflite,
                                      train_manifest, val_manifest,
                                      batch_size, val_cache)
    m_int8 = binary_metrics(int8_probs, labels, 0.5) if int8_probs is not None else {}

    # ---- record ----
    record = {
        "experiment": exp_id,
        "architecture": arch,
        "input_size": image_size,
        "parameters": params,
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "epochs_run": len(history.history["loss"]),
        "val_loss_best": round(float(min(history.history["val_loss"])), 4),
        "val_accuracy": m50["accuracy"],
        "val_fire_recall": m50["fire_recall"],
        "val_fire_precision": m50["fire_precision"],
        "val_fire_f1": m50["fire_f1"],
        "val_fn": m50["fn"],
        "val_fp": m50["fp"],
        "float_tflite_size_bytes": len(float_bytes),
        "int8_tflite_size_bytes": int8_tflite.stat().st_size,
        "int8_val_accuracy": m_int8.get("accuracy", None),
        "int8_val_fire_recall": m_int8.get("fire_recall", None),
        "training_seconds": round(time.time() - t0, 1),
        "seed": seed,
    }
    _append_csv(COMPARISON_CSV, record)
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    # save history csv for plots
    hist_rows = [dict({"epoch": i + 1}, **{k: round(v[i], 5) for k, v in history.history.items()})
                 for i in range(len(history.history["loss"]))]
    with open(exp_dir / "history.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(hist_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(hist_rows)

    model.save(exp_dir / "final.keras")
    best.save(exp_dir / "best.keras")

    print(f"  params={params} val_acc={m50['accuracy']} "
          f"fire_recall={m50['fire_recall']} fire_f1={m50['fire_f1']} "
          f"fn={m50['fn']} int8_acc={m_int8.get('accuracy')} "
          f"float={len(float_bytes)}B int8={int8_tflite.stat().st_size}B"
          f" ({time.time() - t0:.0f}s)")
    return record


# ---------------------------------------------------------------------------
# INT8 quantization helper (lightweight, used for candidate comparison)
# ---------------------------------------------------------------------------
def _make_int8(model, image_size, out_path, train_manifest, val_manifest,
               batch_size, val_cache, rep_size=300):
    """Quantize `model` to full INT8 and return (accuracy, probs) on val.

    Uses the exact same tf.data preprocessing (decode -> bilinear resize ->
    /255) as training, so the INT8 comparison is not polluted by resizing
    differences between libraries.
    """
    import random
    import tensorflow as tf

    h, w = image_size
    rng = random.Random(42)
    rows = _manifest_rows(train_manifest)
    sample = rng.sample(rows, min(rep_size, len(rows)))

    def _decode(p):
        bits = tf.io.read_file(p)
        img = tf.io.decode_image(
            bits, channels=3, dtype=tf.uint8, expand_animations=False
        )
        img = tf.ensure_shape(img, [None, None, 3])
        img = tf.image.resize(img, [h, w], method=tf.image.ResizeMethod.BILINEAR)
        return img

    def rep_ds():
        paths = [str(r["path"]) for r in sample]
        labels = [int(r["label"]) for r in sample]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, l: (_decode(p), l), num_parallel_calls=4)
        ds = ds.map(lambda img, l: (tf.cast(img, tf.float32) / 255.0, l))
        for img, _ in ds.batch(1):
            yield [img.numpy()]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_bytes = converter.convert()
    except Exception as exc:  # noqa: BLE001
        print(f"    int8 conversion failed: {exc}")
        return None, None
    out_path.write_bytes(tflite_bytes)

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = in_d["quantization"]
    out_scale, out_zp = out_d["quantization"]

    # evaluate over the SAME preprocessed validation pipeline
    ds = build_tf_dataset(val_manifest, (h, w), batch_size, train=False,
                          cache_dir=val_cache)
    probs, labels = [], []
    for xb, yb in ds:
        for xi, yi in zip(xb.numpy(), yb.numpy()):
            q = np.round(xi / in_scale + in_zp).astype(np.int8)
            interp.set_tensor(in_d["index"], q[None, ...])
            interp.invoke()
            out = interp.get_tensor(out_d["index"])[0, 0]
            prob = (out.astype(np.float32) - out_zp) * out_scale
            probs.append(prob)
            labels.append(int(yi))
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    return binary_metrics(probs, labels, 0.5), probs


def _manifest_rows(manifest):
    from _helpers import load_manifest
    return load_manifest(manifest)


def _labels_from_manifest(manifest):
    from _helpers import load_manifest
    rows = load_manifest(manifest)
    return [r["path"] for r in rows], [int(r["label"]) for r in rows]


def _append_csv(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(record.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single experiment id")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--arch-size", type=int, default=TRANSFER_IMAGE_SIZE,
                    help="input size for the architecture sweep")
    ap.add_argument("--reuse-caches", action="store_true",
                    help="reuse / force building Preprocessed caches")
    args = ap.parse_args()

    train_manifest = SPLITS_DIR / "train.manifest.csv"
    val_manifest = SPLITS_DIR / "validation.manifest.csv"
    assert train_manifest.exists(), "run 03_create_splits.py first"

    exp_ids = []
    if args.only:
        exp_ids = [args.only]
    else:
        for size in RESOLUTION_CANDIDATES:
            exp_ids.append(f"tiny_cnn_{size}")
        for arch in ARCHITECTURES:
            if arch == "tiny_cnn":
                continue
            exp_ids.append(f"{arch}_{args.arch_size}")
        if DO_TRANSFER_LEARNING and not args.skip_transfer:
            exp_ids.append(f"mobilenet_v2_{TRANSFER_IMAGE_SIZE}")

    print(f"=== GridGuard Fire AI | Step 5: training {len(exp_ids)} experiments ===")
    for exp_id in exp_ids:
        parts = exp_id.rsplit("_", 1)
        arch = "_".join(parts[:-1])
        size = int(parts[-1])
        run_experiment(
            exp_id=exp_id, arch=arch, image_size=size, batch_size=args.batch,
            epochs=args.epochs, train_manifest=train_manifest,
            val_manifest=val_manifest, seed=42,
        )
    print("\nAll experiments complete. Run 06_evaluate_models.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())