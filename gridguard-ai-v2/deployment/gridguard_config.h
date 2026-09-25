#pragma once

// GridGuard AI v2 - ESP32-S3 configuration
// Scaling + quantization parameters extracted from the training pipeline.
// Normalization: x_norm[i] = (x_raw[i] - GG_MEAN[i]) / GG_SCALE[i]  (i = 0..39)
// Then quantize: q = round(x_norm / GG_INPUT_SCALE) + GG_INPUT_ZERO_POINT

#define GG_WINDOW_LEN                 20
#define GG_NUM_INPUTS                 40
#define GG_NUM_OUTPUTS                5
#define GG_INPUT_DTYPE                "int8"
#define GG_OUTPUT_DTYPE               "int8"

static const char* const GG_CLASS_NAMES[GG_NUM_OUTPUTS] = { "NORMAL_OPERATION", "TEMPERATURE_RISING", "THERMAL_DRIFT", "RAPID_TEMPERATURE_RISE", "THERMAL_INSTABILITY" };

// Model I/O quantization (from gridguard_v2_int8.tflite)
const float GG_INPUT_SCALE   = 0.019969076f;
const int   GG_INPUT_ZERO_POINT = -11;
const float GG_OUTPUT_SCALE  = 0.003906250f;
const int   GG_OUTPUT_ZERO_POINT = -128;

// Per-feature z-score normalization (fitted on training data only)
const float GG_MEAN[GG_NUM_INPUTS] = {
    28.601986f, 28.603289f, 28.601723f, 28.603170f, 28.602758f, 28.602478f, 28.601837f, 28.602873f,
    28.604221f, 28.604137f, 28.602839f, 28.604830f, 28.603363f, 28.604498f, 28.604952f, 28.604137f,
    28.605444f, 28.603933f, 28.603600f, 28.604366f, 28.595818f, 28.683590f, 28.760159f, 28.841888f,
    28.933498f, 29.028839f, 29.146078f, 29.234022f, 29.360380f, 29.486721f, 29.618330f, 29.766321f,
    29.913250f, 30.086294f, 30.265520f, 30.474264f, 30.675463f, 30.905836f, 31.168823f, 31.469404f,
};

const float GG_SCALE[GG_NUM_INPUTS] = {
    7.773907f, 7.774244f, 7.774560f, 7.774583f, 7.773906f, 7.773936f, 7.772557f, 7.775130f,
    7.774681f, 7.775038f, 7.775976f, 7.774219f, 7.776169f, 7.774678f, 7.774638f, 7.775418f,
    7.776045f, 7.774603f, 7.775603f, 7.776095f, 7.793808f, 7.785006f, 7.779929f, 7.800858f,
    7.814318f, 7.821912f, 7.841582f, 7.856530f, 7.882776f, 7.919965f, 7.948986f, 7.991180f,
    8.041846f, 8.098342f, 8.170032f, 8.241743f, 8.340005f, 8.455289f, 8.616075f, 8.831120f,
};
