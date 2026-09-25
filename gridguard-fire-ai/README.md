# GridGuard AI — Fire Detection TinyML Model (v1)

**IEEE HART HardwAIre Challenge** project sub-model.

Binary image classifier that detects **visible fire/flames** from an
**AI-Thinker ESP32-CAM (OV2640)**, with an `UNCERTAIN` runtime state derived
from inference confidence (not a third training class).

```
Class 0 -> NO_FIRE
Class 1 -> FIRE
```

```
FIRE probability >= FIRE_THRESHOLD   -> FIRE_DETECTED
FIRE probability in  (NO_FIRE_CUTOFF, FIRE_THRESHOLD) -> UNCERTAIN
FIRE probability <= NO_FIRE_CUTOFF   -> NO_FIRE
```

---

## Results (final model)

_To be filled after the pipeline runs. See `models/final/final_model_config.json`
for the authoritative record._

| Metric | Float | INT8 |
|---|---|---|
| Test accuracy | — | — |
| FIRE recall | — | — |
| FIRE precision | — | — |
| F1 | — | — |
| False negatives (FIRE -> NO_FIRE) | — | — |
| Model size | — | — B |
| Input size | — | — |

Recommended thresholds: `FIRE_THRESHOLD = _.__`, `NO_FIRE_CUTOFF = _.__`.

---

## Pipeline

```
Existing fire dataset
  -> 01 inspect (corruption/duplicates/dimensions/leakage risks)
  -> 02 clean (corrupted exclusion, duplicate grouping)
  -> 03 leakage-safe split (70/15/15, per duplicate-group)
  -> 04 visualize dataset
  -> 05 train experiments (resolution sweep 64/96/128 + 4 architectures
         + transfer-learning reference)
  -> 06 evaluate & select (accuracy + FIRE recall + size + INT8 robustness)
         + threshold optimization (validation only)
  -> 07 hard-case / false-alarm evaluation
  -> 08 float32 TFLite conversion + parity check
  -> 09 INT8 quantization (training-data-only calibration)
  -> 10 quantized model evaluation (float vs INT8)
  -> 11 ESP32-CAM deployment pack + preprocessing verification
```

## Repository structure

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

Key directories:

| Path | Purpose |
|---|---|
| `data/splits/` | train/validation/test **manifests** (70/15/15, leakage-safe) |
| `data/reports/` | inspection, cleaning, corruption & duplicate reports |
| `models/experiments/` | every candidate + comparison table |
| `models/final/` | the selected model + config + quantization params |
| `deployment/` | ESP32-CAM headers, TFLite binaries, deployment guide |
| `results/` | figures, confusion matrices, reports, hard-case analysis |

## How to run

```bash
# from gridguard-fire-ai/ using the repo venv (Python 3.12 + TensorFlow 2.21)
python scripts/01_inspect_dataset.py
python scripts/02_clean_dataset.py
python scripts/03_create_splits.py            # add --materialize for folder trees
python scripts/04_visualize_dataset.py
python scripts/05_train_models.py             # full experiment suite (long on CPU)
python scripts/06_evaluate_models.py          # selects final model + thresholds
python scripts/07_test_hard_cases.py
python scripts/08_convert_tflite.py
python scripts/09_quantize_int8.py
python scripts/10_evaluate_quantized_model.py
python scripts/11_prepare_esp32_deployment.py
```

Reusable tricks:

```bash
python scripts/05_train_models.py --only tiny_cnn_96   # single experiment
python scripts/05_train_models.py --skip-transfer      # skip MobileNetV2 reference
python deployment/verify_esp32_preprocessing.py        # re-verify ESP32 maths
```

## Design decisions

- **Image size** chosen as the smallest input with strong, robust FIRE
  detection (see step 6).  Larger inputs cost ESP32 RAM/FLASH/compute.
- **Training classes**: binary only. `UNCERTAIN` is a *runtime state*
  produced by calibrated thresholds, so it never dilutes the learning signal.
- **FIRE recall first**: a missed fire is the worst failure.  All selection,
  thresholds and quantization checks are recall-conscious.
- **Hard negatives**: bright lamps, sunsets, red/orange objects, reflections
  etc. are analysed in step 7 to test whether the model sees *flames* rather
  than *orange/bright pixels*.
- **INT8 is the deployed model**: the float model is the reference; the INT8
  model must preserve FIRE recall or it is not deployed.

## Notes

- The pre-existing temperature anomaly models (`gridguard-ai/`,
  `gridguard-ai-v2/`) are left untouched.
- The original fire dataset is never modified; `data/splits/*.manifest.csv`
  reference the raw files instead of copying ~1.9 GB.

## License

MIT (see repository root).