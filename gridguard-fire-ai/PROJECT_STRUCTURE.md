# GridGuard Fire AI - Project Structure

```
gridguard-fire-ai/
├── README.md                     # project overview, pipeline, final results
├── PROJECT_STRUCTURE.md          # this file
├── requirements.txt              # Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                      # README: raw source is used in place, unmodified
│   ├── processed/                # README + reserved caches (real caches in $TEMP)
│   ├── splits/                   # train/validation/test manifests (metadata-first)
│   └── reports/                  # inspection/cleaning/duplicate reports
│
├── notebooks/                    # analysis notebooks (thresholds, hard cases)
│
├── scripts/                      # numbered, run in order (see README)
│   ├── config.py                 # single source of truth (paths/hyperparams)
│   ├── _helpers.py               # shared data + preprocessing utilities
│   ├── 01_inspect_dataset.py
│   ├── 02_clean_dataset.py
│   ├── 03_create_splits.py
│   ├── 04_visualize_dataset.py
│   ├── 05_train_models.py
│   ├── 06_evaluate_models.py
│   ├── 07_test_hard_cases.py
│   ├── 08_convert_tflite.py
│   ├── 09_quantize_int8.py
│   ├── 10_evaluate_quantized_model.py
│   └── 11_prepare_esp32_deployment.py
│
├── models/
│   ├── checkpoints/              # per-experiment best-checkpoints
│   ├── experiments/              # per-experiment artifacts + comparison table
│   ├── keras/                    # selected Keras model
│   ├── tflite/                   # generated float32 and INT8 models
│   └── final/                    # final config and evaluation artifacts
│
├── deployment/                   # ESP32-CAM integration pack
│   ├── gridguard_fire_model.h            # INT8 model byte array
│   ├── gridguard_fire_config.h           # all firmware constants
│   ├── gridguard_fire_model.tflite       # float32 reference
│   ├── gridguard_fire_model_int8.tflite  # deployed INT8 model
│   ├── ESP32CAM_DEPLOYMENT.md            # exact inference/preprocessing contract
│   └── verify_esp32_preprocessing.py     # Python simulation of ESP32 maths
│
└── results/
    ├── figures/                  # training/validation/threshold figures
    ├── confusion_matrices/
    ├── classification_reports/
    └── hard_case_results/        # FP/FN analysis + representative images
```

## Principles

1. Raw data is only ever read - never modified.
2. Generated and processed data is separated from raw data.
3. Scripts are numbered by execution order; config lives in one module.
4. Models (training artifacts) are separated from results (evaluation).
5. Deployment files are isolated in `deployment/`.
6. Everything is reproducible: fixed seeds, recorded ratios, kept reports.
7. Test data is never used for tuning, thresholds or quantization calibration.