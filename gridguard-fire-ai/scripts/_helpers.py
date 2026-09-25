"""GridGuard Fire AI - shared data/processing helpers.

Everything in this module is used by multiple pipeline scripts so that the
preprocessing contract (decode -> square resize bilinear -> RGB -> /255) is
defined exactly once and reused during training, TFLite evaluation, INT8
calibration and the ESP32 path verification.
"""
from __future__ import annotations

import io
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import keras
from PIL import Image, ImageOps

from config import DIR_TO_LABEL, SUPPORTED_EXTENSIONS


@keras.saving.register_keras_serializable("GridGuardFire")
class FocalBinaryCrossentropy(keras.losses.Loss):
    """Focal binary cross entropy - focuses learning on FIRE (class 1).

    alpha up-weights positive-class examples; gamma de-emphasizes confident
    negatives so the classifier does not settle into 'always no-fire'.
    Registered as a Keras loss object so saved models reload cleanly.
    """

    def __init__(self, gamma=1.5, alpha=0.6, name="focal_binary_crossentropy",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        return -weight * tf.pow(1.0 - pt, self.gamma) * bce

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def find_images(root: Path) -> Dict[str, List[Path]]:
    """Return {folder_name: [image paths]} discovered under root."""
    found: Dict[str, List[Path]] = {}
    if not root.exists():
        return found
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name not in DIR_TO_LABEL:
            continue
        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        found[folder.name] = files
    return found


def load_image_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Image verify
# ---------------------------------------------------------------------------
def verify_image(path: Path) -> Optional[dict]:
    """Return metadata about an image, or None if it cannot be read at all."""
    data = load_image_bytes(path)
    if data is None:
        return {"corrupt": True, "reason": "unreadable file"}
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
            fmt = im.format
    except Exception as exc:  # noqa: BLE001 - any decode failure = corrupt
        return {"corrupt": True, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "corrupt": False,
        "width": int(w),
        "height": int(h),
        "mode": mode,
        "format": fmt,
        "file_size": len(data),
    }


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Perceptual hash (for near-duplicate detection)
# ---------------------------------------------------------------------------
def dhash_int(path: Path, bits: int = 16) -> Optional[int]:
    """Return a `bits`x`bits` dHash as an int, or None on decode failure.

    Differences between horizontally-adjacent pixels at a reduced size are
    binarised - robust to mild rescale / JPEG / re-encode changes.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("L").resize(
                (bits + 1, bits), Image.LANCZOS
            )
            arr = np.asarray(im, dtype=np.int16)
    except Exception:  # noqa: BLE001
        return None
    diff = arr[:, 1:] > arr[:, :-1]
    rows = [int("".join("1" if b else "0" for b in row), 2) for row in diff]
    return int("".join(f"{r:0{bits}b}" for r in rows), 2)


def hamming_int(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Color statistics (hard-case / false-positive analysis)
# ---------------------------------------------------------------------------
def image_color_stats(path: Path) -> Optional[Dict[str, float]]:
    """Cheap per-image RGB statistics from a 32x32 thumbnail."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((32, 32), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    redness = (r - b) / 255.0
    return {
        "r": float(r.mean() / 255.0),
        "g": float(g.mean() / 255.0),
        "b": float(b.mean() / 255.0),
        "brightness": float(brightness.mean()),
        "redness": float(redness.mean()),
        "redness_std": float(redness.std()),
    }


# ---------------------------------------------------------------------------
# Preprocessing contract (exactly replicates the ESP32-CAM maths)
# ---------------------------------------------------------------------------
def preprocess_rgb(pixels: np.ndarray) -> np.ndarray:
    """RGB uint8 [H,W,3] -> float32 [H,W,3] in [0,1].

    No mean/std normalization is used: the model consumes raw channel values
    in [0,1], which maps 1:1 onto the ESP32 input quantizer.
    """
    return pixels.astype(np.float32) / 255.0


def decode_resize_rgb(
    img_bytes: bytes, size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Decode image bytes to RGB uint8 resized (bilinear) to `size`."""
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = im.resize(size, Image.BILINEAR)
    except Exception:  # noqa: BLE001
        return None
    return np.asarray(im, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Manifest loading / saving
# ---------------------------------------------------------------------------
def load_manifest(csv_path: Path) -> List[dict]:
    import csv

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["path"] = Path(row["path"])
            rows.append(row)
    return rows


def save_manifest(csv_path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_paths_and_labels(manifest_csv: Path) -> Tuple[List[str], List[int]]:
    """Return ([absolute path...], [0|1 label...]) from a manifest CSV."""
    rows = load_manifest(manifest_csv)
    paths = [str(r["path"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    return paths, labels


# ---------------------------------------------------------------------------
# tf.data pipelines (training / validation / test) - manifest driven
# ---------------------------------------------------------------------------
def _augment_tensor(img: "tf.Tensor", size: Tuple[int, int]) -> "tf.Tensor":
    """Realistic OV2640-style augmentation applied to uint8 images [0,255].

    All transforms stay moderate so fire texture is never destroyed.
    """
    import tensorflow as tf

    h, w = size
    # zoom + translation via random crop on a slightly-padded canvas
    pad = 8
    big = tf.image.resize(img, [h + 2 * pad, w + 2 * pad], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.random_crop(big, [h, w, 3])

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.25)
    img = tf.image.random_contrast(img, lower=0.75, upper=1.4)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.25)
    # mild blur (Gaussian-ish via mean over small kernels is too costly on CPU;
    # brightness/contrast/saturation already model exposure variation)
    return img


def build_tf_dataset(
    manifest_csv: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    train: bool,
    cache_dir: Optional[Path] = None,
    shuffle_buffer: int = 2048,
):
    """Build a tf.data pipeline reading the exact files in the manifest.

    Labels are taken directly from the manifest (0 = no_fire, 1 = fire), so
    folder names never need to be parsed at runtime.

    Output tensors: float32 [H,W,3] in [0,1], label int32.
    """
    import tensorflow as tf

    h, w = image_size
    paths, labels = load_paths_and_labels(manifest_csv)

    def _decode(file_path: "tf.Tensor") -> "tf.Tensor":
        bits = tf.io.read_file(file_path)
        img = tf.io.decode_image(
            bits, channels=3, dtype=tf.uint8, expand_animations=False
        )
        img = tf.ensure_shape(img, [None, None, 3])
        img = tf.image.resize(img, [h, w], method=tf.image.ResizeMethod.BILINEAR)
        return img

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, l: (_decode(p), l),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if cache_dir is not None:
        ds = ds.cache(str(cache_dir))
    if train:
        if shuffle_buffer:
            ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(
            lambda img, l: (_augment_tensor(img, (h, w)), l),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        ds = ds.map(lambda img, l: (img, l), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(
        lambda img, l: (tf.cast(img, tf.float32) / 255.0, tf.cast(l, tf.int32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=train)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Prediction + metrics
# ---------------------------------------------------------------------------
def predict_probs(
    model,
    manifest_csv: Path,
    image_size: Tuple[int, int],
    batch_size: int = 32,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a Keras/TFLite-compatible model over a whole manifest.

    Returns (fire_probabilities[N], labels[N]).
    `model` is anything with a .predict(dataset) Keras-style API.
    """
    ds = build_tf_dataset(
        manifest_csv, image_size, batch_size, train=False, cache_dir=cache_dir
    )
    probs = np.asarray(model.predict(ds, verbose=0)).reshape(-1)
    labels = np.asarray(load_paths_and_labels(manifest_csv)[1])
    return probs, labels


def binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    """Compute a full binary-metric dict for fire (class 1) at `threshold`."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "accuracy": round(float(accuracy), 4),
        "fire_recall": round(float(recall), 4),
        "fire_precision": round(float(precision), 4),
        "fire_f1": round(float(f1), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)