"""GridGuard Fire AI - Step 7: Hard-case evaluation.

Answers the central question: does the model detect ACTUAL flames, or has it
just learned 'orange/red/bright = fire'?

The test set (untouched by training/tuning) is scored with the final model
and its selected thresholds.  Every no-fire image is categorized by visual
statistics into fire-lookalike groups (sunset, bright lamp, red/orange
object, low/bright, etc.).  False positives and false negatives are ranked by
distance from the decision boundary and copied for manual inspection.

Outputs:
  results/hard_case_results/hard_case_report.csv
  results/hard_case_results/images/            representative FP/FN copies
  results/hard_case_results/hard_case_summary.txt
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf  # noqa: E402

from config import (  # noqa: E402
    CLASS_NAMES,
    FINAL_CONFIG_JSON,
    FINAL_DIR,
    FINAL_MODEL_KERAS,
    HARD_CASE_CSV,
    HARD_CASE_DIR,
    HARD_CASE_COPY_DIR,
    SPLITS_DIR,
)
from _helpers import (  # noqa: E402
    binary_metrics,
    decode_resize_rgb,
    image_color_stats,
    load_manifest,
    preprocess_rgb,
)

TOP_COPY = 24  # number of representative images copied per side


def categorize(color: dict) -> str:
    if color is None:
        return "unknown"
    b, r, g, rd = color["brightness"], color["r"], color["g"], color["redness"]
    if b >= 0.75:
        return "very_bright_overexposed"
    if rd >= 0.28 and 0.25 <= b < 0.75:
        if rd >= 0.45 and r >= 0.6:
            return "red_orange_object"
        return "warm_sunset_or_light"
    if rd >= 0.15 and g < 0.35:
        return "red_light_lamps"
    if rd >= 0.10:
        return "orange_warm_glow"
    if b >= 0.55:
        return "bright_indoor_or_outdoor"
    if b <= 0.18:
        return "dark_lowlight"
    return "generic_scene"


def main():
    print("=== GridGuard Fire AI | Step 7: hard-case evaluation ===")
    cfg = json.loads(FINAL_CONFIG_JSON.read_text(encoding="utf-8"))
    size = cfg["input_size"]
    fire_thr = float(cfg["fire_threshold"])
    no_fire_thr = float(cfg["no_fire_cutoff"])

    model = tf.keras.models.load_model(FINAL_MODEL_KERAS)
    test_rows = load_manifest(SPLITS_DIR / "test.manifest.csv")
    print(f"Test images: {len(test_rows)}")

    rows_out = []
    for i, r in enumerate(test_rows):
        p = Path(r["path"])
        img = decode_resize_rgb(p.read_bytes(), (size, size))
        x = preprocess_rgb(img)[None, ...]
        prob = float(model.predict(x, verbose=0)[0, 0])
        true = int(r["label"])
        color = image_color_stats(p)
        state = "FIRE_DETECTED" if prob >= fire_thr else (
            "UNCERTAIN" if prob > no_fire_thr else "NO_FIRE")
        rows_out.append({
            "image": str(p),
            "true_label": CLASS_NAMES[true],
            "predicted_label": "FIRE" if state == "FIRE_DETECTED" else (
                "UNCERTAIN" if state == "UNCERTAIN" else "NO_FIRE"),
            "fire_probability": round(prob, 4),
            "confidence_state": state,
            "correct": int((state == "FIRE_DETECTED") == (true == 1)),
            "category": categorize(color),
            "brightness": round(color["brightness"], 3) if color else "",
            "redness": round(color["redness"], 3) if color else "",
            "r_mean": round(color["r"], 3) if color else "",
        })
        if (i + 1) % 500 == 0:
            print(f"  scored {i + 1}/{len(test_rows)}")

    _save_csv(rows_out)

    fps = [r for r in rows_out if r["true_label"] == "no_fire" and r["fire_probability"] >= fire_thr]
    fns = [r for r in rows_out if r["true_label"] == "fire" and r["fire_probability"] < fire_thr]
    unc = [r for r in rows_out if r["confidence_state"] == "UNCERTAIN"]

    fps.sort(key=lambda x: -x["fire_probability"])
    fns.sort(key=lambda x: x["fire_probability"])

    HARD_CASE_COPY_DIR.mkdir(parents=True, exist_ok=True)
    _copy_images(fps[:TOP_COPY])
    _copy_images(fns[:TOP_COPY])

    # category analysis of FPs
    from collections import Counter
    fp_cats = Counter(r["category"] for r in fps)
    fn_cats = Counter(r["category"] for r in fns)

    lines = [
        "=" * 70,
        "GRIDGUARD FIRE AI - HARD CASE / FALSE ALARM ANALYSIS",
        "=" * 70,
        f"Fire threshold       : {fire_thr}",
        f"No-fire cutoff       : {no_fire_thr}",
        f"Test images scored   : {len(rows_out)}",
        f"False positives (no_fire->FIRE_DETECTED): {len(fps)}",
        f"False negatives (fire-><FIRE): {len(fns)}",
        f"UNCERTAIN predictions : {len(unc)}",
        "",
        "-- False-positive categories (does it see fire, or just color?) --",
    ]
    for cat, n in fp_cats.most_common():
        lines.append(f"    {cat:<24}: {n}")
    lines.append("")
    lines.append("-- False-negative categories --")
    for cat, n in fn_cats.most_common():
        lines.append(f"    {cat:<24}: {n}")
    lines += ["", "-- Top false positives (fire probability) --"]
    for r in fps[:TOP_COPY]:
        lines.append(f"    {Path(r['image']).name:<24} p={r['fire_probability']:.3f} "
                     f"cat={r['category']}")
    lines += ["", "-- Top false negatives (fire probability) --"]
    for r in fns[:TOP_COPY]:
        lines.append(f"    {Path(r['image']).name:<24} p={r['fire_probability']:.3f} "
                     f"cat={r['category']}")
    lines += ["",
              "Representative images copied to results/hard_case_results/images/",
              "=" * 70,
    ]
    summary_txt = "\n".join(lines)
    (HARD_CASE_DIR / "hard_case_summary.txt").write_text(summary_txt, encoding="utf-8")
    print(summary_txt)
    return 0


def _copy_images(rows):
    for r in rows:
        src = Path(r["image"])
        tag = "fp" if r["true_label"] == "no_fire" else "fn"
        dst = HARD_CASE_COPY_DIR / f"{tag}_{r['fire_probability']:.3f}_{Path(src).name}"
        try:
            shutil.copy2(src, dst)
        except OSError:
            pass


def _save_csv(rows):
    with open(HARD_CASE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())