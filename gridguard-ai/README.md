# GridGuard AI v1 - Temperature Anomaly Detection

End-to-end TinyML pipeline that detects abnormal temperature patterns and
produces an INT8-quantized model ready for ESP32 / microcontroller deployment.

## Pipeline

Run the scripts in order from this directory:

1. **`scripts/generate_dataset.py`** - Generates a synthetic labelled temperature
   dataset (normal vs. anomaly windows) as `data/temperature_dataset.csv`.
2. **`scripts/prepare_data.py`** - Splits the data (70/15/15), standardizes it
   with a train-only fitted scaler, and caches the splits as NumPy arrays.
3. **`scripts/train_model.py`** - Trains a small MLP classifier (20 -> 32 -> 16 -> 1)
   with early stopping -> `models/gridguard_model_v1.keras`.
4. **`scripts/evaluate_model.py`** - Evaluates the Keras model and writes the
   confusion matrix (`models/model1_confusion_matrix.png`).
5. **`scripts/convert_tflite.py`** - Exports a float32 TensorFlow Lite model
   (`models/gridguard_model_v1_float32.tflite`).
6. **`scripts/quantize_int8.py`** - Applies full INT8 post-training quantization
   for MCU deployment (`models/anomaly_model.tflite`).
7. **`scripts/validate_tflite.py` / `scripts/validate_int8.py`** - Verify the
   converted models against the Keras baseline.
8. **`scripts/convert_model_to_header.py`** - Converts
   `models/anomaly_model.tflite` into `models/anomaly_model.h` for direct
   inclusion in ESP32 firmware.
9. **`scripts/export_scaler_for_esp32.py`** - Prints the scaler mean/scale arrays
   to paste into ESP32 code.
10. **`scripts/test_new_sequences.py` / `scripts/inspect_predictions.py`** -
    Inspect predictions on new data and experiment with thresholds
    (`config.ANOMALY_THRESHOLD`).

## Configuration

All paths, window size, and thresholds live in `config.py`. ESP32-specific
settings live in `deployment/model_config.py`. No hardcoded machine-specific
paths are used.

Note: the deployable INT8 model is kept as `models/anomaly_model.tflite`
(named with the ESP32 header `models/anomaly_model.h` in mind).