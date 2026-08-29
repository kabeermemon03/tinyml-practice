# ============================================================
# GRIDGUARD AI
# MODEL 1 - SHARED CONFIGURATION
# Temperature Anomaly Detection
#
# Single source of truth for project paths and model settings.
# Every GridGuard script imports from this module so the whole
# project works regardless of where it is checked out.
# ============================================================

from pathlib import Path

# Project root is derived from this file's location:
# no machine-specific (hardcoded) paths anywhere else.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEPLOY_DIR = PROJECT_ROOT / "deployment"

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
DATASET_FILE = DATA_DIR / "temperature_dataset_v2.csv"

# Preprocessed NumPy arrays (produced by scripts/prepare_data.py)
X_TRAIN_FILE = DATA_DIR / "X_train.npy"
X_VAL_FILE = DATA_DIR / "X_val.npy"
X_TEST_FILE = DATA_DIR / "X_test.npy"
Y_TRAIN_FILE = DATA_DIR / "y_train.npy"
Y_VAL_FILE = DATA_DIR / "y_val.npy"
Y_TEST_FILE = DATA_DIR / "y_test.npy"
SCALER_MEAN_FILE = DATA_DIR / "scaler_mean.npy"
SCALER_SCALE_FILE = DATA_DIR / "scaler_scale.npy"

# ------------------------------------------------------------------
# Model files
# ------------------------------------------------------------------
KERAS_MODEL_FILE = MODEL_DIR / "gridguard_model_v1.keras"
TFLITE_FLOAT32_FILE = MODEL_DIR / "gridguard_model_v1_float32.tflite"
TFLITE_INT8_FILE = MODEL_DIR / "gridguard_model_v1_int8.tflite"

# Evaluation artifacts
CONFUSION_MATRIX_FILE = MODEL_DIR / "model1_confusion_matrix.png"
TEST_PROBABILITIES_FILE = MODEL_DIR / "model1_test_probabilities.npy"
TEST_PREDICTIONS_FILE = MODEL_DIR / "model1_test_predictions.npy"

# ------------------------------------------------------------------
# Model / signal settings
# ------------------------------------------------------------------
WINDOW_SIZE = 20
SAMPLING_TIME = 1.0          # seconds
BASE_TEMPERATURE = 29.0      # °C

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
RANDOM_SEED = 42

# ------------------------------------------------------------------
# Classification thresholds
# ------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.50         # generic normal/anomaly split
ANOMALY_THRESHOLD = 0.45         # deployment decision threshold