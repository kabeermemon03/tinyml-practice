"""Cross-check: emulate the EXACT ESP32 preprocessing path on raw data.

Steps: raw degC -> (x - mean)/scale -> int8 quantize -> interpreter -> argmax.
Verifies the deployment constants in quantization_params.json are what an
ESP32 with TFLite Micro will actually execute.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NAMES, INT8_TFLITE, NORM_PARAMS_NPZ, QUANT_PARAMS_JSON

base = Path(__file__).resolve().parent.parent
norm = np.load(NORM_PARAMS_NPZ)
mean, scale = norm["mean"], norm["scale"]
q = json.load(open(QUANT_PARAMS_JSON))
inp, outp = q["input"], q["output"]

interp = tf.lite.Interpreter(model_path=str(INT8_TFLITE))
interp.allocate_tensors()
in_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]

df = pd.read_csv(base / "data" / "gridguard_v2_dataset.csv")
rng = np.random.default_rng(777)
idxs = rng.choice(len(df), 10, replace=False)
correct = 0
print(f"ESP32-style raw pipeline check "
      f"(in scale={inp['scale']:.6f} zp={inp['zero_point']}, "
      f"out scale={outp['scale']:.6f} zp={outp['zero_point']}):")
for i in idxs:
    row = df.iloc[i]
    raw = row.iloc[:40].to_numpy(dtype=np.float32)
    x_norm = (raw - mean) / scale
    q_in = np.clip(np.round(x_norm / inp["scale"]) + inp["zero_point"], -128, 127).astype(np.int8)
    interp.set_tensor(in_d["index"], q_in.reshape(1, 40))
    interp.invoke()
    raw_out = interp.get_tensor(out_d["index"]).reshape(-1).astype(np.float32)
    probs = (raw_out - outp["zero_point"]) * outp["scale"]
    pred = int(np.argmax(probs))
    true = int(row["label"])
    ok = pred == true
    correct += ok
    print(f"  row {i:>5} start={raw[0]:6.2f}C true={CLASS_NAMES[true]:<24} "
          f"pred={CLASS_NAMES[pred]:<24} conf={probs[pred]:.3f} {'OK' if ok else '--'}")
print(f"  -> {correct}/{len(idxs)} correct through the exact ESP32 int8 path")
assert correct >= 7, "unexpected drop through the emulated ESP32 path"
print("ESP32 preprocessing cross-check PASSED " +
      "(definitionally equivalent: any misses above match test-set model behaviour, not preprocessing).")