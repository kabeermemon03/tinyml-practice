# GridGuard AI v2 - ESP32-S3 Deployment

## Model summary

| Property | Value |
|---|---|
| Model file | `models/gridguard_v2_int8.tflite` |
| Task | 5-class temperature-behaviour classification |
| Input shape | [1 40] (`<class 'numpy.int8'>`) |
| Input scale | 0.019969075918197632 |
| Input zero point | -11 |
| Output shape | [1 5] (`<class 'numpy.int8'>`) |
| Output scale | 0.00390625 |
| Output zero point | -128 |
| Model size | 8912 bytes (INT8), 16644 bytes (float32) |
| Class map | 0=NORMAL_OPERATION, 1=TEMPERATURE_RISING, 2=THERMAL_DRIFT, 3=RAPID_TEMPERATURE_RISE, 4=THERMAL_INSTABILITY |

## Files to use on the ESP32-S3

- `deployment/gridguard_model.h` -> the INT8 TFLite model bytes
- `deployment/gridguard_config.h` -> scaling and quantization constants

## Runtime workflow

### Phase 1 - environmental baseline learning

Disable inference until 20 valid samples are collected. They may be at any
temperature (e.g. 20C, 26C, 31C, 36C) - no fixed baseline is assumed.

```
float baseline_window[20];
for (int i = 0; i < 20; i++) {
    baseline_window[i] = read_ntc_celsius();   // wait one sample period per sample
}
```

### Phase 2 - continuous monitoring

Always keep the latest 20 samples in `current_window`. Each time a new sample
arrives, shift the window and run inference.

```
float current_window[20];
// after each new reading:
memmove(&current_window[0], &current_window[1], 19 * sizeof(float));
current_window[19] = read_ntc_celsius();
```

### Preprocessing (must match training)

1. Concatenate baseline (20 values) then current (20 values) -> raw[40].
2. Per-feature z-score using GG_MEAN / GG_SCALE (from gridguard_config.h):
   `x_norm[i] = (raw[i] - GG_MEAN[i]) / GG_SCALE[i]`
3. Quantize to int8 for the model input:
   `q = round(x_norm / GG_INPUT_SCALE) + GG_INPUT_ZERO_POINT` (clamp to [-128, 127]).
4. Run TFLite Micro inference.
5. Dequantize output: `prob[c] = (out_int8[c] - GG_OUTPUT_ZERO_POINT) * GG_OUTPUT_SCALE`.
6. Class = argmax over 5 probabilities; confidence = max probability.

### TFLite Micro sketch skeleton

```cpp
#include "gridguard_model.h"
#include "gridguard_config.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static tflite::MicroInterpreter* interpreter = nullptr;

void setup_model() {
    static tflite::AllOpsResolver resolver;
    static constexpr tflite::Model* model =
        tflite::GetModel(gridguard_model_tflite);
    static uint8_t arena[64 * 1024];  // works for ~9 KB model
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, arena, sizeof(arena));
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
}

int run_classifier(const float* baseline, const float* current) {
    int8_t* input = interpreter->GetInput(0)->data.int8;
    float raw[GG_NUM_INPUTS];
    for (int i = 0; i < GG_WINDOW_LEN; i++) raw[i] = baseline[i];
    for (int i = 0; i < GG_WINDOW_LEN; i++) raw[i + GG_WINDOW_LEN] = current[i];
    for (int i = 0; i < GG_NUM_INPUTS; i++) {
        float x = (raw[i] - GG_MEAN[i]) / GG_SCALE[i];
        int q = (int)roundf(x / GG_INPUT_SCALE) + GG_INPUT_ZERO_POINT;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input[i] = (int8_t)q;
    }
    interpreter->Invoke();
    const int8_t* out = interpreter->GetOutput(0)->data.int8;
    float probs[GG_NUM_OUTPUTS];
    int best = 0;
    float best_conf = -1.0f;
    for (int c = 0; c < GG_NUM_OUTPUTS; c++) {
        probs[c] = ((float)out[c] - GG_OUTPUT_ZERO_POINT) * GG_OUTPUT_SCALE;
        if (probs[c] > best_conf) { best_conf = probs[c]; best = c; }
    }
    return best;  // plus best_conf for the confidence
}
```

## Notes

- The scaler mean/scale were fitted **only on the training split**. No Leakage.
- The input int8 scale/zero point come from the path-through input quantization
  of `models/gridguard_v2_int8.tflite` (see `models/quantization_params.json`).
- The 20-sample windows are time-ordered: sample 0 is the oldest.
- The representative dataset for INT8 quantization was drawn from the training
  split only (2000 examples).
