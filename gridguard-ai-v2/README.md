# GridGuard AI v2 - TinyML Temperature Behaviour Classifier (ESP32-S3)

A completely rebuilt TinyML pipeline for the IEEE HART project. Trained from
scratch on a new synthetic dataset with no reuse of the previous dataset,
labels, thresholds or training logic.

The ESP32-S3 learns the *environmental baseline* (first 20 samples) and then
judges the *behavior and pattern* of the next 20 samples against it, while
still preserving absolute temperature magnitude in the inputs.

## Classes

| Index | Name | Meaning |
|---|---|---|
| 0 | NORMAL_OPERATION | Stable / naturally varying relative to the learned environment |
| 1 | TEMPERATURE_RISING | Gradual smooth upward trend (not automatically critical) |
| 2 | THERMAL_DRIFT | Slow but persistent movement away from the baseline |
| 3 | RAPID_TEMPERATURE_RISE | Fast / accelerating abnormal rise (high priority) |
| 4 | THERMAL_INSTABILITY | Irregular fluctuations, jumps, oscillations |

## Results

| Metric | Value |
|---|---|
| Dataset | 16,000 rows (3,200 / class), start temp 15C-42C, balanced, seed 42 |
| Split | 70 / 15 / 15 stratified |
| Architecture | Dense 40 -> 48 -> 24 -> 12 -> 5 (softmax) |
| Parameters | 3,509 |
| Test accuracy (Keras) | 0.8879 |
| Test accuracy (float TFLite) | 0.8879 |
| Test accuracy (INT8 TFLite) | 0.8842 |
| Float TFLite size | 16,644 B |
| INT8 TFLite size | 8,912 B |
| Normalization | per-feature z-score, fitted on training split only |
| INT8 input | shape [1,40], scale 0.019969, zero point -11 |
| INT8 output | shape [1,5], scale 0.003906, zero point -128 |

Per-class (test): NORMAL 0.96 F1, RISING 0.82 F1, DRIFT 0.71 F1,
RAPID 0.98 F1, INSTABILITY 0.95 F1. The RISING <-> DRIFT pair shows ~21%
mutual confusion, which is intentional: slow smooth upward drift is
genuinely ambiguous with a gradual rise (see class design in step 1).

## Pipeline (run in order)

```bash
venv workspaces activate, then from gridguard-ai-v2/scripts:

python 01_generate_dataset.py        # dataset CSV + overlap sanity stats
python 02_visualize_dataset.py       # example sequences + overlap histogram
python 03_prepare_data.py            # stratified split + train-only scaler
python 04_experiment_architectures.py# 6 small MLPs compared (acc/size)
python 05_train_model.py             # final model + accuracy/loss curves
python 06_evaluate_model.py          # report, confusion matrix, pair analysis
python 07_manual_scenarios.py        # custom unseen scenarios + confidence
python 08_convert_tflite.py          # float32 TFLite + validation
python 09_quantize_int8.py           # INT8 + rep dataset + accuracy compare
python 10_generate_esp32_files.py    # ESP32-S3 headers + deployment doc
python 11_crosscheck_esp32_path.py   # exact ESP32 preprocessing cross-check
```

## Directory layout

| Path | Contents |
|---|---|
| `data/` | generated CSV, npz splits, normalization params |
| `figures/` | all plots (dataset, training, confusion, accuracy) |
| `models/` | .keras, float32 .tflite, int8 .tflite, quantization params |
| `logs/` | historical CSVs, reports, manual scenario results |
| `deployment/` | `gridguard_model.h`, `gridguard_config.h`, `ESP32_DEPLOYMENT.md` |

## ESP32 preprocessing contract (must match training exactly)

1. Collect 20 baseline samples on power-up (any starting temperature).
2. Keep a rolling window of the latest 20 samples as the current window.
3. Build the 40-value input: baseline[20] then current[20].
4. `x_norm[i] = (x_raw[i] - GG_MEAN[i]) / GG_SCALE[i]` (global z-score, preserves
   absolute temperature - the means live around the true °C).
5. Quantize to the model input: `q = round(x_norm / input_scale) + input_zero_point`
   clamped to int8.
6. Inference, then dequantize the 5 outputs and take the argmax + confidence.

Full sketch skeleton and constants: `deployment/ESP32_DEPLOYMENT.md`.

## Key design decisions

- No fixed baseline temperature: every example starts at a random 15-42C.
- Classes overlap deliberately: slopes, shapes, onsets, amplitudes and noise
  are randomized so a threshold or fixed-slope rule cannot reproduce the model.
- Absolute temperature is preserved by a *global* per-feature z-score fitted
  only on training data (mean ~28.6C baseline zone, scale ~7.8C).
- INT8 quantization used a 2000-example representative set drawn from the
  training split only (no leakage), with int8 input/output for TFLite Micro.