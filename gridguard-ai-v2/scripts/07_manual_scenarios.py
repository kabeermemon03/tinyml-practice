"""Step 7: GridGuard AI v2 - manual unseen-scenario testing.

Although the model was trained on synthetic data, this step deliberately
crafts new, never-before-seen sequences (fresh seeds + hand-built edge cases)
and prints baseline, current, true class, predicted class and confidence.
It also probes absolute-temperature awareness and the tricky class pairs.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_CHECKPOINT,
    CLASS_NAMES,
    MANUAL_SCENARIOS_CSV,
    NORM_PARAMS_NPZ,
    WINDOW_LEN,
)
from _generator_helpers import gen_normal, gen_rising, gen_drift, gen_rapid, gen_instability

W = WINDOW_LEN


def make_scenarios():
    scenarios = []

    def add(name, class_id, builder):
        scenarios.append((name, class_id, builder))

    add("cold normal stable (~20C)", 0, lambda rng: gen_normal(rng, 20.2))
    add("warm normal fluttering (~36C)", 0, lambda rng: gen_normal(rng, 35.8))
    add("high normal tiny ripple (~41C)", 0, lambda rng: gen_normal(rng, 40.9))
    add("slow rise from 20C", 1, lambda rng: gen_rising(rng, 20.4))
    add("moderate rise at high absolute temp (~40C)", 1, lambda rng: gen_rising(rng, 39.7))
    add("very slow upward drift with plateau (ambiguous)", 2, lambda rng: gen_drift(rng, 24.1))
    add("downward drift from 25.6C", 2, lambda rng: gen_drift(rng, 25.6))
    add("wavy persistent drift up from 31C", 2, lambda rng: gen_drift(rng, 31.3))
    add("fast linear rise from 22C", 3, lambda rng: gen_rapid(rng, 22.0))
    add("delayed rapid rise from 30C", 3, lambda rng: gen_rapid(rng, 30.0))
    add("accelerating rapid rise from low temp", 3, lambda rng: gen_rapid(rng, 17.8))
    add("rapid rise at very high absolute temp (~42C)", 3, lambda rng: gen_rapid(rng, 41.5))
    add("small-amplitude thermal instability", 4, lambda rng: gen_instability(rng, 26.3))
    add("large chaotic instability with spike", 4, lambda rng: gen_instability(rng, 32.7))
    add("low-temp oscillating instability (~19C)", 4, lambda rng: gen_instability(rng, 19.4))
    return scenarios


def main():
    print("=" * 72)
    print("GridGuard AI v2 - Step 7: Manual unseen-scenario testing")
    print("=" * 72)
    model = tf.keras.models.load_model(BEST_CHECKPOINT)
    norm = np.load(NORM_PARAMS_NPZ)
    mean, scale = norm["mean"], norm["scale"]

    rows = []
    for scenario_id, (name, class_id, builder) in enumerate(make_scenarios()):
        rng = np.random.default_rng(10_000 + scenario_id)
        baseline, current = builder(rng)
        start = float(baseline[0])
        raw = np.concatenate([baseline, current]).astype(np.float32)
        normed = (raw - mean) / scale
        probs = model.predict(normed.reshape(1, -1), verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        top3 = np.argsort(probs)[::-1][:3]
        correct = "OK " if pred == class_id else "MISS"

        print(f"\n[{scenario_id:02d}] {name}")
        print(f"    start temp ~{start:.1f}C")
        print(f"    baseline: {np.round(baseline, 2).tolist()}")
        print(f"    current : {np.round(current, 2).tolist()}")
        print(f"    true={CLASS_NAMES[class_id]} | pred={CLASS_NAMES[pred]} "
              f"({conf:.3f}) [{correct}]")
        print(f"    top-3   : " + " ".join(
            f"{CLASS_NAMES[i]}={probs[i]:.2f}" for i in top3))

        rows.append([scenario_id, name, round(start, 2), CLASS_NAMES[class_id],
                     CLASS_NAMES[pred], round(conf, 4), "correct" if pred == class_id else "wrong"])

    with open(MANUAL_SCENARIOS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "description", "start_temp", "true_class",
                         "predicted_class", "confidence", "result"])
        writer.writerows(rows)
    print(f"\nManual scenarios saved: {MANUAL_SCENARIOS_CSV}")

    n_ok = sum(1 for r in rows if r[6] == "correct")
    print(f"Summary: {n_ok}/{len(rows)} scenarios classified correctly")


if __name__ == "__main__":
    main()