"""
Step 5: Per-source demographic / imaging-protocol report.

Produces three files:
    report/
      |- demographic_report.md     Markdown table draft for the manuscript
      |- demographic_summary.csv   Machine-readable summary (one row per source)
      `- patient_manifest.csv      Per-subject manifest (one row per subject)

Patient-ID parsing:
  - WJ:           stem with the `_view` suffix removed = patient_id
  - GD/HB/SX/TJ:  stem with both `<prefix>_` and `_view` removed = patient_id
  - VD:           de-identified; subject-level IDs unavailable, so each image
                  is treated as its own independent unit

Per-source statistics reported:
  - Number of images, number of unique subjects
  - Image counts per view (lateral / flexion / extension)
  - Protocol coverage: lateral only / flex+ext only / lateral + flex-ext pair / all three views
  - Images-per-subject distribution (mean +/- SD, median [IQR], range)

Usage:
    python step5_demographics.py \\
        --center_lists ./center_lists \\
        --out_dir      ./report
"""
from __future__ import annotations
import argparse, csv, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median

import numpy as np

SOURCES = ["WJ", "VD", "GD", "HB", "SX", "TJ"]
SOURCE_FULL = {
    "WJ": "Wangjing Hospital (Beijing)",
    "VD": "VinDr-SpineXR (public)",
    "GD": "Guangdong TCM Hospital",
    "HB": "Hebei Univ. Affiliated Hospital",
    "SX": "Shanxi Integrated Medicine Hospital",
    "TJ": "Tianjin TCM Univ. 1st Hospital",
}
SOURCE_PREFIX = {"GD": "gz", "HB": "hb", "SX": "sx", "TJ": "tj"}

VIEW_RX = re.compile(r"_(flexion|lateral|extension)$", re.IGNORECASE)


# --- Patient ID parsing --------------------------------------------------
def parse_stem(img_path: Path, source: str) -> tuple[str, str]:
    """Return (patient_id, view). view in {lateral, flexion, extension}."""
    stem = img_path.stem

    if source == "VD":
        # De-identified — treat each image as an independent unit.
        return stem, "lateral"

    m = VIEW_RX.search(stem)
    view = m.group(1).lower() if m else "lateral"
    stem_wo_view = VIEW_RX.sub("", stem)

    if source in SOURCE_PREFIX:
        pfx = SOURCE_PREFIX[source] + "_"
        if stem_wo_view.lower().startswith(pfx):
            stem_wo_view = stem_wo_view[len(pfx):]

    return stem_wo_view, view


# --- Per-source statistics -----------------------------------------------
def stat_source(source: str, image_paths: list[Path]) -> dict:
    patients: dict[str, set[str]] = defaultdict(set)
    images_per_pt: dict[str, int] = defaultdict(int)

    view_counts = {"lateral": 0, "flexion": 0, "extension": 0}

    for p in image_paths:
        pid, view = parse_stem(p, source)
        patients[pid].add(view)
        images_per_pt[pid] += 1
        if view in view_counts:
            view_counts[view] += 1

    n_pt = len(patients)

    lateral_only = sum(1 for v in patients.values() if v == {"lateral"})
    flex_ext_no_lat = sum(1 for v in patients.values()
                          if v == {"flexion", "extension"})
    lat_plus_pair   = sum(1 for v in patients.values()
                          if {"flexion", "extension"}.issubset(v))
    three_views     = sum(1 for v in patients.values()
                          if v >= {"lateral", "flexion", "extension"})

    counts = list(images_per_pt.values())
    ipp_mean = float(np.mean(counts)) if counts else 0.0
    ipp_sd   = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
    ipp_med  = float(np.median(counts)) if counts else 0.0
    q1, q3   = (float(np.percentile(counts, 25)),
                float(np.percentile(counts, 75))) if counts else (0.0, 0.0)
    ipp_min  = min(counts) if counts else 0
    ipp_max  = max(counts) if counts else 0

    return {
        "source":          source,
        "full_name":       SOURCE_FULL[source],
        "n_images":        len(image_paths),
        "n_patients":      n_pt,
        "lateral_images":  view_counts["lateral"],
        "flexion_images":  view_counts["flexion"],
        "extension_images":view_counts["extension"],
        "lateral_only":    lateral_only,
        "flex_ext_pair":   lat_plus_pair,       # has lateral AND has flex-ext pair
        "three_views":     three_views,         # has all three views
        "flex_ext_no_lat": flex_ext_no_lat,     # flex+ext only (no lateral)
        "ipp_mean":        ipp_mean,
        "ipp_sd":          ipp_sd,
        "ipp_median":      ipp_med,
        "ipp_q1":          q1,
        "ipp_q3":          q3,
        "ipp_min":         ipp_min,
        "ipp_max":         ipp_max,
        "patients":        patients,
        "images_per_pt":   images_per_pt,
    }


# --- Output --------------------------------------------------------------
def write_summary_csv(stats: dict[str, dict], out: Path) -> None:
    cols = [
        "source", "full_name", "n_images", "n_patients",
        "lateral_images", "flexion_images", "extension_images",
        "lateral_only", "flex_ext_pair", "three_views", "flex_ext_no_lat",
        "ipp_mean", "ipp_sd", "ipp_median", "ipp_q1", "ipp_q3",
        "ipp_min", "ipp_max",
    ]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in SOURCES:
            w.writerow([stats[c][k] for k in cols])
    print(f"  wrote {out}")


def write_manifest_csv(stats: dict[str, dict], out: Path) -> None:
    cols = ["source", "patient_id", "n_images",
            "has_lateral", "has_flexion", "has_extension"]
    rows = []
    for c in SOURCES:
        for pid, views in stats[c]["patients"].items():
            rows.append([
                c, pid, stats[c]["images_per_pt"][pid],
                int("lateral"   in views),
                int("flexion"   in views),
                int("extension" in views),
            ])
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"  wrote {out}  ({len(rows)} rows)")


def fmt_ipp(s: dict) -> str:
    """Image-per-patient summary string."""
    if s["n_patients"] == 0:
        return "—"
    return (f"{s['ipp_mean']:.2f} ± {s['ipp_sd']:.2f}  "
            f"(median {s['ipp_median']:.0f} "
            f"[{s['ipp_q1']:.0f}–{s['ipp_q3']:.0f}], "
            f"range {s['ipp_min']}–{s['ipp_max']})")


def write_markdown(stats: dict[str, dict], out: Path) -> None:
    total_img = sum(stats[c]["n_images"]   for c in SOURCES)
    total_pt  = sum(stats[c]["n_patients"] for c in SOURCES)
    total_lat = sum(stats[c]["lateral_images"]   for c in SOURCES)
    total_flx = sum(stats[c]["flexion_images"]   for c in SOURCES)
    total_ext = sum(stats[c]["extension_images"] for c in SOURCES)
    total_lat_only = sum(stats[c]["lateral_only"] for c in SOURCES)
    total_flxext   = sum(stats[c]["flex_ext_pair"]  for c in SOURCES)
    total_3view    = sum(stats[c]["three_views"]    for c in SOURCES)

    lines: list[str] = []
    add = lines.append

    add("# Clinical Sources Demographic & Imaging Protocol Report")
    add("")
    add(f"*Generated: {datetime.now():%Y-%m-%d %H:%M}*")
    add("")
    add(f"**Total cohort**: {total_img} radiographs from "
        f"{total_pt} subjects across 6 sources.")
    add("")
    add("---")
    add("")
    add("## Table 1. Imaging inventory by source")
    add("")
    add("| Source | Full name | Images | Subjects | Lateral | Flexion | Extension |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for c in SOURCES:
        s = stats[c]
        nsub = (f"{s['n_patients']}*" if c == "VD" else f"{s['n_patients']}")
        add(f"| {c} | {s['full_name']} | {s['n_images']} | {nsub} | "
            f"{s['lateral_images']} | {s['flexion_images']} | {s['extension_images']} |")
    add(f"| **Total** | — | **{total_img}** | **{total_pt}** | "
        f"**{total_lat}** | **{total_flx}** | **{total_ext}** |")
    add("")
    add("*VinDr-SpineXR is de-identified; subject-level identifiers are "
        "unavailable, so images are counted as independent units.")
    add("")
    add("---")
    add("")
    add("## Table 2. Imaging-protocol coverage (subjects)")
    add("")
    add("| Source | Lateral only | Lateral + Flex+Ext pair | All three views | "
        "Flex+Ext only (no lateral) |")
    add("|---|---:|---:|---:|---:|")
    for c in SOURCES:
        s = stats[c]
        add(f"| {c} | {s['lateral_only']} | {s['flex_ext_pair']} | "
            f"{s['three_views']} | {s['flex_ext_no_lat']} |")
    add(f"| **Total** | **{total_lat_only}** | **{total_flxext}** | "
        f"**{total_3view}** | — |")
    add("")
    add(f"> **{total_flxext}** subjects have matched flexion–extension pairs "
        "and are eligible for dynamic-parameter analysis (Section 3.5).")
    add("")
    add("---")
    add("")
    add("## Table 3. Images-per-subject distribution")
    add("")
    add("| Source | mean ± SD | median [IQR] | range |")
    add("|---|---|---|---|")
    for c in SOURCES:
        s = stats[c]
        add(f"| {c} | {s['ipp_mean']:.2f} ± {s['ipp_sd']:.2f} | "
            f"{s['ipp_median']:.0f} [{s['ipp_q1']:.0f}–{s['ipp_q3']:.0f}] | "
            f"{s['ipp_min']}–{s['ipp_max']} |")
    add("")
    add("---")
    add("")
    add("## Notes for manuscript")
    add("")
    add("- **Demographic metadata** (age, sex, diagnosis) was not available at "
        "data-freeze time; Table 1 reports imaging inventory only.")
    add("- **VinDr-SpineXR** is a public anonymized dataset without subject-level IDs; "
        "each radiograph is treated as an independent unit throughout downstream analyses.")
    add("- **Dynamic-parameter analysis** (Cobb ROM, disc height index) requires "
        f"matched flex–ext pairs; the {total_flxext} eligible subjects are a "
        "subset of the four clinical sources that acquired functional views.")
    add("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out}")


# --- main ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center_lists", required=True)
    ap.add_argument("--out_dir",      required=True)
    args = ap.parse_args()

    list_dir = Path(args.center_lists)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== step 5: demographics report ===\n")

    stats: dict[str, dict] = {}
    for c in SOURCES:
        lst = list_dir / f"{c}_images.txt"
        paths = [Path(ln.strip()) for ln in lst.read_text().splitlines() if ln.strip()]
        stats[c] = stat_source(c, paths)
        s = stats[c]
        print(f"  {c}  images={s['n_images']:5d}  subjects={s['n_patients']:5d}  "
              f"[L={s['lateral_images']} F={s['flexion_images']} "
              f"E={s['extension_images']}]  "
              f"3-view={s['three_views']}  flex-ext-pair={s['flex_ext_pair']}")

    print()
    write_summary_csv (stats, out_dir / "demographic_summary.csv")
    write_manifest_csv(stats, out_dir / "patient_manifest.csv")
    write_markdown    (stats, out_dir / "demographic_report.md")

    print(f"\nDone. Open {out_dir/'demographic_report.md'} to view the report.")


if __name__ == "__main__":
    main()
