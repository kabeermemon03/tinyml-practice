"""GridGuard Fire AI - Step 2: Dataset cleaning & validation.

Builds the *cleaned* manifest used by every downstream step.  It:
  * drops corrupted / unreadable images (reported in step 1),
  * tags every exact-duplicate group with a stable group id,
  * marks one canonical representative per group (so duplicate copies never
    leak across train/test),
  * reports near-duplicate pairs (from step 1) for human review,
  * detects cross-class duplicate groups (obvious labeling errors).

It NEVER modifies, moves or deletes the original dataset.

Outputs:
  data/reports/clean_manifest.csv     all valid images + duplicate group ids
  data/reports/cleaning_summary.json  cleaning statistics
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CLEAN_MANIFEST,
    CORRUPTED_CSV,
    CLASS_NAMES,
    DIR_TO_LABEL,
    RAW_MANIFEST,
    REPORTS_DIR,
)
from _helpers import load_image_bytes, verify_image  # noqa: E402

CLEANING_SUMMARY_JSON = REPORTS_DIR / "cleaning_summary.json"


def main():
    print("=== GridGuard Fire AI | Step 2: dataset cleaning ===")
    raw_rows = _load_csv(RAW_MANIFEST)
    corrupt_rows = _load_csv(CORRUPTED_CSV)
    print(f"Raw valid images       : {len(raw_rows)}")
    print(f"Corrupted (excluded)   : {len(corrupt_rows)}")

    corrupt_paths = {row["path"] for row in corrupt_rows}

    # Re-verify anything left (double guard) and drop stragglers.
    droppable = set()
    for row in raw_rows:
        p = Path(row["path"])
        if str(p) in corrupt_paths:
            droppable.add(str(p))
            continue
        meta = verify_image(p)
        if meta is None or meta.get("corrupt"):
            droppable.add(str(p))

    valid_rows = [r for r in raw_rows if r["path"] not in droppable]
    print(f"Valid after re-verify  : {len(valid_rows)} (dropped {len(droppable)})")

    # ---- exact-duplicate groups (md5) => assigning a shared group id ----
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in valid_rows:
        groups[r["md5"]].append(r)
    dup_groups = {m: v for m, v in groups.items() if len(v) > 1}

    group_ids = {}
    for gid, (md5, members) in enumerate(dup_groups.items(), 1):
        for r in members:
            group_ids[id(r)] = gid

    # Representative = first member per group (stable ordering from step 1)
    reps = {gid: members[0]["path"] for gid, (_, members) in enumerate(dup_groups.items(), 1)}

    rows_out = []
    n_rep = 0
    for r in valid_rows:
        rid = id(r)
        gid = group_ids.get(rid, 0)
        is_rep = (gid == 0) or (r["path"] == reps[gid])
        if is_rep:
            n_rep += 1
        rows_out.append({
            **{k: r.get(k, "") for k in r.keys()},
            "duplicate_group_id": gid,
            "is_representative": int(is_rep),
        })
        r["duplicate_group_id"] = gid
        r["is_representative"] = int(is_rep)

    # ---- cross-class duplicate groups (labeling risk) ----
    cross_class = []
    for gid, members in dup_groups.items():
        lbls = {r["label"] for r in members}
        if len(lbls) > 1:
            cross_class.append({
                "group_id": gid,
                "md5": gid,
                "classes": [CLASS_NAMES[int(l)] for l in sorted(lbls)],
                "paths": [r["path"] for r in members],
            })

    _save_manifest(CLEAN_MANIFEST, rows_out)

    # ---- near-duplicate stats from step 1 ----
    near_pairs = _load_csv(REPORTS_DIR / "near_duplicate_pairs.csv")
    cross_class_near = sum(
        1 for a, b in ((Path(r["path_a"]), Path(r["path_b"])) for r in near_pairs)
        if DIR_TO_LABEL[a.parent.name] != DIR_TO_LABEL[b.parent.name]
    )

    summary = {
        "raw_valid_images": len(raw_rows),
        "corrupted_excluded": len(corrupt_rows),
        "additional_dropped_on_recheck": len(droppable),
        "clean_images": len(rows_out),
        "unique_representative_units": n_rep,
        "exact_duplicate_groups": len(dup_groups),
        "duplicate_copies_kept_for_split": sum(len(v) - 1 for v in dup_groups.values()),
        "cross_class_duplicate_groups": len(cross_class),
        "near_duplicate_pairs_reported": len(near_pairs),
        "near_duplicate_cross_class_pairs": cross_class_near,
        "class_clean_counts": {
            CLASS_NAMES[i]: sum(1 for r in rows_out if r["label"] == str(i))
            for i in range(len(CLASS_NAMES))
        },
    }
    with open(CLEANING_SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nClean images          : {summary['clean_images']}")
    for name, cnt in summary["class_clean_counts"].items():
        print(f"  {name}: {cnt}")
    print(f"Representative units : {summary['unique_representative_units']}")
    print(f"Cross-class dup groups (labeling risk): {summary['cross_class_duplicate_groups']}")
    if cross_class:
        print("Cross-class groups:")
        for c in cross_class[:20]:
            print(f"  group {c['group_id']}: {c['classes']} -> {Path(c['paths'][0]).name}")
    print(f"\nCleaned manifest     : {CLEAN_MANIFEST}")
    print(f"Summary              : {CLEANING_SUMMARY_JSON}")
    return 0


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save_manifest(csv_path, rows):
    fieldnames = ["path", "folder", "class", "label", "md5", "width", "height",
                  "mode", "format", "file_size", "aspect_ratio", "brightness",
                  "redness", "r_mean", "g_mean", "b_mean", "dhash",
                  "duplicate_group_id", "is_representative"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())