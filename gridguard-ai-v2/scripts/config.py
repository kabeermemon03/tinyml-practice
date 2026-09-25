"""GridGuard AI v2 - central configuration.

All shared paths, class names and hyper-parameters live here so every
script in the pipeline stays consistent and reproducible.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
DEPLOYMENT_DIR = BASE_DIR / "deployment"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

for d in (DATA_DIR, FIGURES_DIR, MODELS_DIR, DEPLOYMENT_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SAMPLE_RATE_SECONDS = 1.0

WINDOW_LEN = 20
NUM_FEATURES = WINDOW_LEN * 2
NUM_CLASSES = 5

EXAMPLES_PER_CLASS = 3200
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

START_TEMP_MIN = 15.0
START_TEMP_MAX = 42.0

CLASS_NAMES = {
    0: "NORMAL_OPERATION",
    1: "TEMPERATURE_RISING",
    2: "THERMAL_DRIFT",
    3: "RAPID_TEMPERATURE_RISE",
    4: "THERMAL_INSTABILITY",
}
CLASS_NAMES_LIST = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

BASE_COLUMNS = [f"baseline_{i+1}" for i in range(WINDOW_LEN)]
CURRENT_COLUMNS = [f"current_{i+1}" for i in range(WINDOW_LEN)]
FEATURE_COLUMNS = BASE_COLUMNS + CURRENT_COLUMNS

DATASET_CSV = DATA_DIR / "gridguard_v2_dataset.csv"
DATASET_META_JSON = DATA_DIR / "dataset_meta.json"
DATASET_OVERVIEW_PNG = FIGURES_DIR / "dataset_class_examples.png"

NORM_PARAMS_NPZ = DATA_DIR / "normalization_params.npz"
NORM_PARAMS_JSON = DATA_DIR / "normalization_params.json"

TRAIN_NPZ = DATA_DIR / "train.npz"
VAL_NPZ = DATA_DIR / "val.npz"
TEST_NPZ = DATA_DIR / "test.npz"

ARCH_EXPERIMENTS_CSV = LOGS_DIR / "architecture_experiments.csv"
ARCH_COMPARISON_PNG = FIGURES_DIR / "architecture_comparison.png"

FINAL_MODEL_KERAS = MODELS_DIR / "gridguard_v2_final.keras"
BEST_CHECKPOINT = MODELS_DIR / "gridguard_v2_best.keras"
HISTORY_CSV = LOGS_DIR / "training_history.csv"
ACC_LOSS_PNG = FIGURES_DIR / "training_accuracy_loss.png"

CONFUSION_PNG = FIGURES_DIR / "confusion_matrix.png"
REPORT_CSV = LOGS_DIR / "classification_report.csv"
PAIR_ANALYSIS_CSV = LOGS_DIR / "pair_confusion_analysis.csv"

MANUAL_SCENARIOS_CSV = LOGS_DIR / "manual_scenarios.csv"

FLOAT_TFLITE = MODELS_DIR / "gridguard_v2_float32.tflite"
INT8_TFLITE = MODELS_DIR / "gridguard_v2_int8.tflite"
FLOAT_VS_INT8_PNG = FIGURES_DIR / "float_vs_int8_accuracy.png"
QUANT_PARAMS_JSON = MODELS_DIR / "quantization_params.json"

MODEL_HEADER_H = DEPLOYMENT_DIR / "gridguard_model.h"
CONFIG_HEADER_H = DEPLOYMENT_DIR / "gridguard_config.h"
DEPLOYMENT_README = DEPLOYMENT_DIR / "ESP32_DEPLOYMENT.md"
PIPELINE_README = BASE_DIR / "README.md"

TRAIN_EPOCHS = 80
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3

FINAL_ARCHITECTURE = (48, 24, 12)