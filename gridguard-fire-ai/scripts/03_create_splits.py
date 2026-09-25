"""GridGuard Fire AI - Step 3: Leakage-safe dataset splitting.

Creates the reproducible 70 / 15 / 15 stratified split.

Leakage protection:
  * The split unit is the *duplicate group*, not the individual image.
    Exact-duplicate copies (and their group members) always land in the SAME
    split, so identical pixels can never sit in both train and test.
  * Cross-class duplicate groups stay indivisible as well.
  * Splits are stratified per class to preserve the ~62/38 class balance.

By default the split is expressed as MANIFESTS (CSV) that reference the raw
files, avoiding a 1.9GB duplicate copy.  Pass `--materialize` to physically
copy files into data/splits/<split>/<class>/ folder trees.

Outputs:
  data/splits/train.manifest.csv      (also validation.manifest.csv, test)
  data/splits/splits_manifest.csv     every image + assigned split
  data/splits/split_summary.json      ratios + per-split class counts
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CLEAN_MANIFEST,
    LABEL_TO_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    SPLITS_MANIFEST,
    SPLIT_SUMMARY_JSON,
    SPLIT_NAMES,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from _helpers import load_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="GridGuard Fire AI dataset split")
    parser.add_argument("--materialize", action="store_true",
                        help="copy images into data/splits/<split>/<class>/ trees")
    args = parser.parse_args()

    print("=== GridGuard Fire AI | Step 3: leakage-safe split ===")
    rows = load_manifest(CLEAN_MANIFEST)
    if not rows:
        print("FATAL: clean manifest empty - run 02_clean_dataset.py first")
        return 1
    print(f"Clean images        : {len(rows)}")

    # ------------------------- build indivisible units --------------------
    units: dict[int, list[dict]] = {}          # unit_id -> member rows
    for r in rows:
        gid = int(r.get("duplicate_group_id", 0) or 0)
        uid = gid if gid > 0 else _fresh_id(units)
        units.setdefault(uid, []).append(r)

    # unit representative = first row (already in stable step-1 order)
    unit_list = []
    for uid, members in units.items():
        members.sort(key=lambda m: m["path"])
        rep = members[0]
        unit_list.append({
            "unit_id": uid,
            "label": int(rep["label"]),
            "files": members,
        })
    print(f"Indivisible units   : {len(unit_list)}")

    # ------------------- exclude contradictory (cross-class) units ----------
    # Identical-pixel duplicates labeled BOTH fire and no_fire are labeling
    # errors.  They are excluded from every split (never removed from raw).
    kept, excluded = [], []
    for u in unit_list:
        lbls = {int(m["label"]) for m in u["files"]}
        if len(lbls) > 1:
            excluded.append(u)
        else:
            kept.append(u)
    if excluded:
        _save_conflicts(SPLITS_DIR / "conflicting_label_groups.csv", excluded)
        print(f"Excluded contradictory units: {len(excluded)} "
              f"({sum(len(u['files']) for u in excluded)} images) -> "
              f"conflicting_label_groups.csv")
    unit_list = kept

    # --------------------------- stratified split -------------------------
    rng = random.Random(RANDOM_SEED)
    per_class = defaultdict(list)
    for u in unit_list:
        per_class[u["label"]].append(u)

    assignment = {}
    for lbl, us in per_class.items():
        us.sort(key=lambda u: u["unit_id"])
        rng.shuffle(us)
        n = len(us)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        # n_test = n - n_train - n_val (respect requested ratios as closely
        # as integer counts allow, always exhausting all units).
        for i, u in enumerate(us):
            if i < n_train:
                assignment[u["unit_id"]] = "train"
            elif i < n_train + n_val:
                assignment[u["unit_id"]] = "validation"
            else:
                assignment[u["unit_id"]] = "test"

    # --------------------------------- write ------------------------------
    out_rows = []
    for u in unit_list:
        split = assignment[u["unit_id"]]
        for m in u["files"]:
            out_rows.append({**{k: m.get(k, "") for k in m}, "split": split})

    _save(splits_manifest=SPLITS_MANIFEST, rows=out_rows)

    for split in SPLIT_NAMES:
        path = SPLITS_DIR / f"{split}.manifest.csv"
        _save(splits_manifest=path, rows=[r for r in out_rows if r["split"] == split])

    if args.materialize:
        print("Materializing folder trees...")
        _materialize(out_rows)

    summary = _summarize(out_rows)
    with open(SPLIT_SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSplit ratios target : {TRAIN_RATIO} / {VAL_RATIO} / {TEST_RATIO}")
    for split in SPLIT_NAMES:
        row = summary["splits"][split]
        print(f"  {split:<10} images={row['images']:>6} "
              f"fire={row['fire']:>5} no_fire={row['no_fire']:>5}")
    print(f"\nManifests written to {SPLITS_DIR}")
    return 0


_fresh_counter = [0]

def _fresh_id(units):
    _fresh_counter[0] -= 1
    return _fresh_counter[0]


def _save_conflicts(path, units):
    import csv as _csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["unit_id", "labels", "paths"])
        for u in units:
            w.writerow([u["unit_id"],
                        ",".join(sorted({str(int(m["label"])) for m in u["files"]})),
                        "; ".join(str(m["path"]) for m in u["files"])])


def _save(splits_manifest, rows):
    fieldnames = ["path", "folder", "class", "label", "md5", "width", "height",
                  "mode", "format", "file_size", "aspect_ratio", "brightness",
                  "redness", "r_mean", "g_mean", "b_mean", "dhash",
                  "duplicate_group_id", "is_representative", "split"]
    parent = splits_manifest.parent
    parent.mkdir(parents=True, exist_ok=True)
    with open(splits_manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _materialize(rows):
    from config import RAW_DATASET_ROOT

    for r in rows:
        src = Path(r["path"])
        rel = src.relative_to(RAW_DATASET_ROOT)
        folder = src.parent.name
        flat = "_".join(rel.parts[:-1] + (rel.name,)) if len(rel.parts) > 1 else rel.name
        dst = SPLITS_DIR / r["split"] / (LABEL_TO_DIR[int(r["label"])][0]) / flat
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)


def _summarize(rows):
    counts = {"train": {"fire": 0, "no_fire": 0, "images": 0},
              "validation": {"fire": 0, "no_fire": 0, "images": 0},
              "test": {"fire": 0, "no_fire": 0, "images": 0}}
    for r in rows:
        s = counts[r["split"]]
        s["images"] += 1
        if int(r["label"]) == 1:
            s["fire"] += 1
        else:
            s["no_fire"] += 1
    return {
        "seed": RANDOM_SEED,
        "ratios": {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "splits": counts,
    }


if __name__ == "__main__":
    raise SystemExit(main())