"""GridGuard Fire AI - Step 1: Dataset inspection & validation.

Read-only analysis of the raw fire dataset.  Nothing is modified.

Outputs (reports):
  data/reports/dataset_summary.json          overall + per-class summary
  data/reports/class_distribution.csv        image counts by class
  data/reports/corrupted_images.csv          every file that failed to decode
  data/reports/duplicate_analysis.csv        exact-duplicate statistics
  data/reports/duplicate_groups.json         md5 -> [identical file paths]
  data/reports/near_duplicate_pairs.csv      perceptual-hash near duplicates
  data/reports/raw_manifest.csv              every valid image: path/class/md5/
                                             dims/mode/color stats (metadata)
  data/reports/dataset_inspection_report.txt human-readable report
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CLASS_DISTRIBUTION_CSV,
    CLASS_NAMES,
    CORRUPTED_CSV,
    DATASET_SUMMARY_JSON,
    DIR_TO_LABEL,
    DUPLICATE_ANALYSIS_CSV,
    DUPLICATE_GROUPS_JSON,
    NEAR_DUPLICATE_CSV,
    PHASH_BITS,
    PHASH_HAMMING_NEAR,
    RANDOM_SEED,
    RANDOM_SEED as SEED,
    RAW_DATASET_ROOT,
    RAW_MANIFEST,
    REPORTS_DIR,
)
from _helpers import (  # noqa: E402
    dhash_int,
    find_images,
    hamming_int,
    image_color_stats,
    load_image_bytes,
    md5_of,
    verify_image,
)

REPORT_TXT = REPORTS_DIR / "dataset_inspection_report.txt"


def folder_to_label(folder: str) -> int:
    return DIR_TO_LABEL[folder]


def corrupt_row(p: Path, folder: str, reason: str) -> dict:
    return {"path": str(p), "folder": folder, "reason": reason}


def find_near_duplicates(hash_cache, prefix_bits=24):
    """Group near-duplicate images by bucketed dHash similarity.

    Images are placed into buckets by the top `prefix_bits` of their 16x16
    dHash (256 bits).  Only comparisons inside a bucket are performed, which
    keeps this near O(N) instead of O(N^2).  Returns sorted list of
    (path_a, path_b, hamming_distance) tuples <= PHASH_HAMMING_NEAR.
    """
    buckets = defaultdict(list)
    for path, hval in hash_cache.items():
        if hval is None:
            continue
        bucket = hval >> (PHASH_BITS * PHASH_BITS - prefix_bits)
        buckets[bucket].append((path, hval))

    pairs = set()
    for items in buckets.values():
        items.sort(key=lambda t: t[1])
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                p1, h1 = items[i]
                p2, h2 = items[j]
                d = hamming_int(h1, h2)
                if d <= PHASH_HAMMING_NEAR:
                    key = (p1, p2) if str(p1) < str(p2) else (p2, p1)
                    pairs.add((key[0], key[1], d))
    return sorted(pairs, key=lambda t: -t[2])


def main():
    print("=== GridGuard Fire AI | Step 1: dataset inspection ===")
    print(f"Raw dataset root   : {RAW_DATASET_ROOT}")
    if not RAW_DATASET_ROOT.exists():
        print(f"FATAL: dataset not found: {RAW_DATASET_ROOT}")
        return 1

    found = find_images(RAW_DATASET_ROOT)
    print("Discovered classes:", {k: len(v) for k, v in found.items()})

    start = time.time()
    corrupted: list[dict] = []
    rows: list[dict] = []
    class_total = {DIR_TO_LABEL[f]: 0 for f in found}
    class_corrupt = {DIR_TO_LABEL[f]: 0 for f in found}
    fmt_counter: Counter = Counter()
    mode_counter: Counter = Counter()
    dim_counter: Counter = Counter()
    aspect_counter: Counter = Counter()
    hash_cache: dict[Path, int | None] = {}

    all_files = [(folder, p) for folder, files in found.items() for p in files]
    total = len(all_files)
    PROBE_EVERY = 2500

    for n, (folder, p) in enumerate(all_files, 1):
        lbl = DIR_TO_LABEL[folder]
        class_total[lbl] += 1

        raw = load_image_bytes(p)
        meta = verify_image(p)
        if raw is None or meta is None or meta.get("corrupt"):
            reason = (meta or {}).get("reason", "unreadable file")
            corrupted.append(corrupt_row(p, folder, reason))
            class_corrupt[lbl] += 1
            hash_cache[p] = None
            continue

        md5 = md5_of(p)
        w, h = meta["width"], meta["height"]
        dim_counter[(w, h)] += 1
        mode_counter[meta["mode"]] += 1
        fmt_counter[meta["format"] or "?"] += 1
        aspect = w / h
        aspect_counter[round(aspect, 2)] += 1
        color = image_color_stats(p) or {}
        hash_cache[p] = dhash_int(p, bits=PHASH_BITS)

        rows.append({
            "path": str(p),
            "folder": folder,
            "class": CLASS_NAMES[lbl],
            "label": lbl,
            "md5": md5,
            "width": w,
            "height": h,
            "mode": meta["mode"],
            "format": meta["format"] or "?",
            "file_size": meta["file_size"],
            "aspect_ratio": round(aspect, 4),
            "brightness": round(color.get("brightness", 0.0), 4),
            "redness": round(color.get("redness", 0.0), 4),
            "r_mean": round(color.get("r", 0.0), 4),
            "g_mean": round(color.get("g", 0.0), 4),
            "b_mean": round(color.get("b", 0.0), 4),
            "dhash": hash_cache[p] if hash_cache[p] is not None else "",
        })
        if n % PROBE_EVERY == 0:
            print(f"  probed {n}/{total} ({time.time() - start:.0f}s)")

    # ------------------------------------------------------- exact duplicates
    print("Detecting exact duplicates (MD5)...")
    md5_groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        md5_groups[r["md5"]].append(r["path"])
    dup_groups = {k: v for k, v in md5_groups.items() if len(v) > 1}

    dup_stats = {
        "total_images_checked": len(rows),
        "unique_md5": len(md5_groups),
        "exact_duplicate_groups": len(dup_groups),
        "duplicate_copies_beyond_first": sum(len(v) - 1 for v in dup_groups.values()),
        "cross_class_duplicate_groups": _cross_class_dup_groups(dup_groups),
    }

    # -------------------------------------------------- near duplicate search
    print("Detecting near-duplicates (bucketed dHash)...")
    near_pairs = find_near_duplicates(hash_cache)

    # --------------------------------------------------------------- writes
    _save_manifest(RAW_MANIFEST, rows)
    _write_csv(CORRUPTED_CSV, ["path", "folder", "reason"],
               [{"path": c["path"], "folder": c["folder"], "reason": c["reason"]} for c in corrupted])
    _write_csv(DUPLICATE_ANALYSIS_CSV, ["md5", "count", "paths"],
               [{"md5": m, "count": len(v), "paths": "; ".join(v)}
                for m, v in sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))])
    with open(DUPLICATE_GROUPS_JSON, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in dup_groups.items()}, fh, indent=2)
    _write_csv(NEAR_DUPLICATE_CSV, ["path_a", "path_b", "hamming"],
               [{"path_a": a, "path_b": b, "hamming": d} for a, b, d in near_pairs])

    valid = {lbl: class_total[lbl] - class_corrupt[lbl] for lbl in class_total}
    summary = {
        "dataset_root": str(RAW_DATASET_ROOT),
        "total_files_found": total,
        "total_corrupted": len(corrupted),
        "total_valid": len(rows),
        "classes": {
            CLASS_NAMES[lbl]: {
                "label_index": lbl,
                "valid": valid[lbl],
                "corrupt": class_corrupt[lbl],
                "total_found": class_total[lbl],
            }
            for lbl in sorted(class_total)
        },
        "imbalance_fire_over_no_fire": round(
            valid[1] / max(1, valid[0]), 4),
        "formats": dict(fmt_counter),
        "modes": dict(mode_counter),
        "most_common_dimensions": dim_counter.most_common(15),
        "most_common_aspect_ratios": aspect_counter.most_common(15),
        "duplicates": dup_stats,
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_threshold_hamming": PHASH_HAMMING_NEAR,
        "inspection_time_seconds": round(time.time() - start, 1),
        "seed": SEED,
    }
    _write_json(DATASET_SUMMARY_JSON, summary)
    _write_csv(CLASS_DISTRIBUTION_CSV, ["class", "label_index", "valid", "corrupt", "total_found"],
               [{"class": CLASS_NAMES[lbl], "label_index": lbl,
                 "valid": valid[lbl], "corrupt": class_corrupt[lbl],
                 "total_found": class_total[lbl]} for lbl in sorted(class_total)])

    report_lines = _build_report_text(summary, corrupted, near_pairs, dup_groups)
    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nReports written to {REPORTS_DIR}")
    return 0


def _build_report_text(summary, corrupted, near_pairs, dup_groups):
    lines = [
        "=" * 72,
        "GRIDGUARD FIRE AI - DATASET INSPECTION REPORT",
        "=" * 72,
        f"Dataset root : {summary['dataset_root']}",
        f"Seed         : {summary['seed']}",
        "",
        "-- Class inventory --",
    ]
    for name, c in summary["classes"].items():
        lines.append(
            f"  {name:<10} (label {c['label_index']}): valid={c['valid']} corrupt={c['corrupt']} "
            f"total={c['total_found']}"
        )
    lines += [
        "",
        f"-- Image formats --  {summary['formats']}",
        f"-- Image modes --    {summary['modes']}",
        "",
        "-- Most common dimensions (w,h): count --",
    ]
    for (w, h), c in summary["most_common_dimensions"]:
        lines.append(f"    {w}x{h}: {c}")
    lines += ["", "-- Exact duplicates (MD5) --",
              f"  images checked           : {summary['duplicates']['total_images_checked']}",
              f"  unique md5               : {summary['duplicates']['unique_md5']}",
              f"  duplicate groups         : {summary['duplicates']['exact_duplicate_groups']}",
              f"  copies beyond the first  : {summary['duplicates']['duplicate_copies_beyond_first']}",
              f"  CROSS-CLASS groups (risk): {summary['duplicates']['cross_class_duplicate_groups']}",
              "",
              "-- Near duplicates (dHash) --",
              f"  pairs found (hamming <= {summary['near_duplicate_threshold_hamming']}): {len(near_pairs)}",
              "  top-10 by distance:",
    ]
    for a, b, d in near_pairs[:10]:
        lines.append(f"    {Path(a).name}  <->  {Path(b).name}  (hamming {d}, classes "
                     f"{Path(a).parent.name}/{Path(b).parent.name})")
    lines += ["", "-- Corrupted images --", f"  {len(corrupted)} recorded in corrupted_images.csv"]
    for c in corrupted[:10]:
        lines.append(f"    {c['path']}  ({c['reason']})")
    lines += [
        "",
        "-- Imbalance --",
        f"  valid fire / no_fire ratio: {summary['imbalance_fire_over_no_fire']}",
        "",
        "Note: 'not_fire' (9755) outnumbers 'fire' (5853). Stratified splits",
        "and class-weighting will be used downstream.  The RAM dataset was not",
        "modified.  Full metadata is in raw_manifest.csv.",
        "=" * 72,
    ]
    return lines


def _cross_class_dup_groups(dup_groups):
    n = 0
    for paths in dup_groups.values():
        labels = {DIR_TO_LABEL[Path(p).parent.name]
                  for p in paths if Path(p).parent.name in DIR_TO_LABEL}
        if len(labels) > 1:
            n += 1
    return n


def _save_manifest(csv_path, rows):
    fieldnames = ["path", "folder", "class", "label", "md5", "width", "height",
                  "mode", "format", "file_size", "aspect_ratio", "brightness",
                  "redness", "r_mean", "g_mean", "b_mean", "dhash"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_csv(csv_path, fieldnames, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())