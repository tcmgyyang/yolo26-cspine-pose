"""
Step 0: Data cleaning, source-of-origin identification, layout unification, and
per-source image-list generation.

Performs four tasks:
  1) Remove samples in extracted/ with empty label files (failed detections)
     along with the corresponding images.
  2) Symlink images and labels from extracted/ + extracted_labels/ into
     images/extracted/ and labels/extracted/. YOLO derives label paths by
     replacing "/images/" with "/labels/" in the image path, so the two
     directory trees must mirror each other.
  3) Identify the clinical source for each image based on its path/filename:
       - In images/{train,val,test}/:
           * contains "_flexion" / "_lateral" / "_extension"  -> WJ
           * otherwise                                        -> VD
       - In images/extracted/:
           * gz_*  -> GD       * hb_* -> HB
           * sx_*  -> SX       * tj_* -> TJ
  4) Write one image-path list per source: center_lists/{XX}_images.txt.

Usage:
    python step0_prepare_data.py --data_root ./dataset
"""
from __future__ import annotations
import argparse, os, re, shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PREFIX_MAP = {"gz": "GD", "hb": "HB", "sx": "SX", "tj": "TJ"}
VIEW_TOKENS = ("_flexion", "_lateral", "_extension")


# --- 1) Remove failed-detection samples in extracted/ ---------------------
def cleanup_failed(data_root: Path) -> int:
    img_dir = data_root / "extracted"
    lbl_dir = data_root / "extracted_labels"
    removed = 0
    for lbl in lbl_dir.glob("*.txt"):
        if lbl.stat().st_size == 0:
            stem = lbl.stem
            for ext in IMG_EXTS:
                img = img_dir / f"{stem}{ext}"
                if img.exists():
                    img.unlink()
                    break
            lbl.unlink()
            removed += 1
            print(f"  [rm] {stem}")
    print(f"[cleanup] removed {removed} empty-label samples")
    return removed


# --- 2) Mirror extracted/ into images/extracted + labels/extracted --------
def unify_structure(data_root: Path) -> None:
    src_img = data_root / "extracted"
    src_lbl = data_root / "extracted_labels"
    dst_img = data_root / "images"  / "extracted"
    dst_lbl = data_root / "labels"  / "extracted"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    # Symlink images (fall back to copy when symlinks are not permitted)
    linked = 0
    for f in src_img.iterdir():
        if f.suffix.lower() not in IMG_EXTS:
            continue
        link = dst_img / f.name
        if not link.exists():
            try:
                os.symlink(f.resolve(), link)
            except OSError:
                shutil.copy2(f, link)
            linked += 1
    print(f"[link] images/extracted  <- {linked} new links")

    # Symlink labels
    linked = 0
    for f in src_lbl.iterdir():
        if f.suffix.lower() != ".txt":
            continue
        link = dst_lbl / f.name
        if not link.exists():
            try:
                os.symlink(f.resolve(), link)
            except OSError:
                shutil.copy2(f, link)
            linked += 1
    print(f"[link] labels/extracted  <- {linked} new links")


# --- 3) Source-of-origin classification -----------------------------------
def classify_image(img_path: Path, data_root: Path) -> str | None:
    rel = img_path.relative_to(data_root).as_posix()
    name = img_path.name.lower()

    # GD/HB/SX/TJ: images/extracted/<prefix>_...
    if rel.startswith("images/extracted/"):
        m = re.match(r"([a-z]+)_", name)
        if m and m.group(1) in PREFIX_MAP:
            return PREFIX_MAP[m.group(1)]
        return None

    # WJ / VD: images/{train,val,test}/
    if rel.startswith(("images/train/", "images/val/", "images/test/")):
        if any(t in name for t in VIEW_TOKENS):
            return "WJ"
        return "VD"

    return None


# --- 4) Write per-source image lists --------------------------------------
def build_center_lists(data_root: Path, out_dir: Path) -> dict[str, list[Path]]:
    centers: dict[str, list[Path]] = defaultdict(list)

    # Walk the four candidate image directories
    search_dirs = [
        data_root / "images" / "train",
        data_root / "images" / "val",
        data_root / "images" / "test",
        data_root / "images" / "extracted",
    ]
    for d in search_dirs:
        if not d.exists():
            print(f"  [warn] {d} not found, skipped")
            continue
        for img in d.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            c = classify_image(img, data_root)
            if c is None:
                continue
            centers[c].append(img.resolve())

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[center stats]")
    for c in ["WJ", "VD", "GD", "HB", "SX", "TJ"]:
        paths = sorted(centers.get(c, []))
        out_file = out_dir / f"{c}_images.txt"
        out_file.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
        print(f"  {c}: {len(paths):5d} images  ->  {out_file}")

    return centers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="e.g. ./dataset")
    ap.add_argument("--out_dir",   default="./center_lists")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    assert root.exists(), f"data_root not found: {root}"

    print(f"=== step 0: data preparation @ {root} ===\n")

    cleanup_failed(root)
    print()
    unify_structure(root)
    print()
    build_center_lists(root, Path(args.out_dir))

    print("\nDone. Next: python step1_make_loso_splits.py --center_lists ./center_lists")


if __name__ == "__main__":
    main()
