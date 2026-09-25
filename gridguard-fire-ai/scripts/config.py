"""GridGuard Fire AI v1 - central configuration.

Single source of truth for paths, classes, split ratios, training
hyper-parameters, thresholds and the ESP32-CAM deployment contract.

The original raw fire dataset is NEVER modified.  It lives in the existing
`gridguard-fire-ai/fire-dataset/fire_dataset` directory and is referenced
(read-only) from here.  Every script in this pipeline imports this module,
so settings stay consistent and reproducible across machines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = DATA_DIR / "reports"

MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
FINAL_DIR = MODELS_DIR / "final"
KERAS_DIR = MODELS_DIR / "keras"
TFLITE_DIR = MODELS_DIR / "tflite"

DEPLOYMENT_DIR = BASE_DIR / "deployment"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"
HARD_CASE_DIR = RESULTS_DIR / "hard_case_results"

SCRIPTS_DIR = BASE_DIR / "scripts"

# TensorFlow dataset caches live OUTSIDE the repository (OS temp dir) so the
# OneDrive-synced repo never has to upload hundreds of MB of cache blobs.
# Caches are rebuilt automatically and are not part of the committed project.
CACHE_ROOT = Path(os.environ.get("TEMP", str(PROCESSED_DIR))) / "gridguard_fire_ai_cache"

for _d in (
    DATA_DIR, RAW_DIR, PROCESSED_DIR, SPLITS_DIR, REPORTS_DIR,
    MODELS_DIR, CHECKPOINTS_DIR, EXPERIMENTS_DIR, FINAL_DIR, KERAS_DIR,
    TFLITE_DIR,
    DEPLOYMENT_DIR, RESULTS_DIR, FIGURES_DIR, CONFUSION_DIR,
    CLASSIFICATION_REPORTS_DIR, HARD_CASE_DIR,
    PROCESSED_DIR / "cached_tensors", CACHE_ROOT,
):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw dataset location (read-only source, never modified)
# ---------------------------------------------------------------------------
# The original fire dataset shipped with this repository lives at:
#   <repo>/gridguard-fire-ai/fire-dataset/fire_dataset/{fire,not_fire}
RAW_DATASET_ROOT = BASE_DIR / "fire-dataset" / "fire_dataset"

RAW_SOURCE_README = RAW_DIR / "README.md"
RAW_MANIFEST = REPORTS_DIR / "raw_manifest.csv"
CLEAN_MANIFEST = REPORTS_DIR / "clean_manifest.csv"
CORRUPTED_CSV = REPORTS_DIR / "corrupted_images.csv"
DUPLICATE_ANALYSIS_CSV = REPORTS_DIR / "duplicate_analysis.csv"
DUPLICATE_GROUPS_JSON = REPORTS_DIR / "duplicate_groups.json"
NEAR_DUPLICATE_CSV = REPORTS_DIR / "near_duplicate_pairs.csv"
DATASET_SUMMARY_JSON = REPORTS_DIR / "dataset_summary.json"
CLASS_DISTRIBUTION_CSV = REPORTS_DIR / "class_distribution.csv"

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Class 0 -> NO_FIRE, Class 1 -> FIRE.  Directory names must map to labels.
LABEL_TO_DIR = {0: ["no_fire", "not_fire"], 1: ["fire"]}
DIR_TO_LABEL = {d: lbl for lbl, dirs in LABEL_TO_DIR.items() for d in dirs}

CLASS_NAMES = ["NO_FIRE", "FIRE"]
NUM_CLASSES = 2

# ---------------------------------------------------------------------------
# Dataset cleaning / split
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_NAMES = ("train", "validation", "test")

# Perceptual-hash parameters for near-duplicate detection
PHASH_BITS = 16          # produces a 16x16 -> 256-bit hash
PHASH_HAMMING_NEAR = 12  # hamming distance <= this counts as "near duplicate"

SPLITS_MANIFEST = SPLITS_DIR / "splits_manifest.csv"
SPLIT_SUMMARY_JSON = SPLITS_DIR / "split_summary.json"

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
INPUT_CHANNELS = 3
BATCH_SIZE = 32
TRAIN_EPOCHS = 45
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 1.5        # down-weights easy negatives; boosts FIRE recall

# Candidate input resolutions (square) evaluated against ESP32-CAM budgets
RESOLUTION_CANDIDATES: List[int] = [64, 96, 128]

# Architectures to compare (see 05_train_models.py for definitions)
ARCHITECTURES: List[str] = ["tiny_cnn", "deep_cnn", "separable_cnn", "tinymobilenet"]

# Transfer-learning reference model (NOT a deployment candidate by default)
TRANSFER_ARCH = "mobilenet_v2"   # reference only
DO_TRANSFER_LEARNING = True
TRANSFER_IMAGE_SIZE = 96

# Experiment run identifiers
EXP_META_JSON = EXPERIMENTS_DIR / "experiments_meta.json"
COMPARISON_CSV = EXPERIMENTS_DIR / "model_comparison_validation.csv"

# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------
FINAL_MODEL_KERAS = KERAS_DIR / "gridguard_fire_best.keras"
FINAL_HISTORY_PNG = FIGURES_DIR / "final_training_accuracy_loss.png"

FINAL_CONFIG_JSON = FINAL_DIR / "final_model_config.json"
THRESHOLD_REPORT_JSON = RESULTS_DIR / "threshold_report.json"

# Threshold / uncertainty (optimized later from validation - see 06)
FIRE_THRESHOLD = 0.50          # provisional; overwritten by threshold analysis
UNCERTAIN_LOW = 0.40           # provisional; overwritten by threshold analysis

# ---------------------------------------------------------------------------
# TFLite / INT8
# ---------------------------------------------------------------------------
REP_DATASET_SIZE = 600         # representative calibration images (train only)
QUANTIZATION_ATTEMPTS = 3      # retries with growing rep set on recall loss

FLOAT_TFLITE = TFLITE_DIR / "gridguard_fire_model_float32.tflite"
INT8_TFLITE = TFLITE_DIR / "gridguard_fire_model_int8.tflite"
QUANT_PARAMS_JSON = FINAL_DIR / "quantization_params.json"

DEPLOY_FLOAT_TFLITE = DEPLOYMENT_DIR / "gridguard_fire_model.tflite"
DEPLOY_INT8_TFLITE = DEPLOYMENT_DIR / "gridguard_fire_model_int8.tflite"
DEPLOY_MODEL_HEADER = DEPLOYMENT_DIR / "gridguard_fire_model.h"
DEPLOY_CONFIG_HEADER = DEPLOYMENT_DIR / "gridguard_fire_config.h"
DEPLOY_README = DEPLOYMENT_DIR / "ESP32CAM_DEPLOYMENT.md"
DEPLOY_VERIFY_SCRIPT = DEPLOYMENT_DIR / "verify_esp32_preprocessing.py"
DEPLOY_VERIFY_CSV = RESULTS_DIR / "esp32_path_verification.csv"

FLOAT_VS_INT8_CSV = RESULTS_DIR / "float_vs_int8_comparison.csv"

# ---------------------------------------------------------------------------
# Hard-case analysis
# ---------------------------------------------------------------------------
HARD_CASE_CSV = HARD_CASE_DIR / "hard_case_report.csv"
HARD_CASE_COPY_DIR = HARD_CASE_DIR / "images"

# ---------------------------------------------------------------------------
# ESP32-CAM contract (documented & enforced in 11_prepare_esp32_deployment.py)
# ---------------------------------------------------------------------------
ESP32_FRAMESIZE_HINT = "FRAMESIZE_96X96"   # closest OV2640 native size; see deployment doc
ESP32_OUTPUT_FORMAT = "GRAYSCALE_DISABLED" # model uses RGB; do NOT use grayscale

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_final_config() -> dict:
    """Serialize the final, reproducibility-critical configuration."""
    return {
        "classes": CLASS_NAMES,
        "input_size": int(FINAL_INPUT_SIZE[0]),
        "input_channels": INPUT_CHANNELS,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "epochs": TRAIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "focal_gamma": FOCAL_GAMMA,
        "fire_threshold": FIRE_THRESHOLD,
        "uncertain_low": UNCERTAIN_LOW,
        "preprocessing": {
            "decode": "JPEG/NP decode",
            "resize": "bilinear to INPUT_SIZE",
            "channels": "RGB",
            "normalize": "x / 255.0 -> [0,1]",
        },
    }


# INPUT_SIZE is finalized after the resolution experiment (05 -> 06).  It is
# persisted to FINAL_EXP_META_JSON and re-read everywhere else; a dynamic
# default of 96 keeps scripts runnable before that step exists.
FINAL_INPUT_SIZE = [96, 96]


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_final_input_size(size: Tuple[int, int]) -> None:
    global FINAL_INPUT_SIZE
    FINAL_INPUT_SIZE = list(size)