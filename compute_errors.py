"""
Compute per-keypoint Euclidean localization error (pixels and mm) by pairing
YOLO-pose prediction txt files with ground-truth YOLO-pose txt files, then
reading DICOM PixelSpacing to convert px -> mm.

Output:  errors_long.csv  with columns
         image_id, vertebra, keypoint, err_px, err_mm, mm_per_px,
         pred_x, pred_y, gt_x, gt_y, visible_gt, visible_pred

Usage:
    python compute_errors.py \
        --pred_dir   /path/to/pred_txt \
        --gt_dir     /path/to/gt_txt   \
        --img_dir    /path/to/images   \
        --out_csv    /path/to/errors_long.csv \
        [--mm_per_px 0.148]     # optional global override (e.g. scanner-wide)
        [--fallback_mm_per_px 0.148]   # used only when DICOM PixelSpacing missing
"""
from __future__ import annotations
import argparse, csv, math, re, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kp_schema import FLAT_KP, KPRecord


def infer_view_from_name(stem: str) -> str:
    """Rule (user-defined):
         name contains 'extension' -> Extension
         name contains 'flexion'   -> Flexion
         otherwise                 -> Neutral  (lateral)
    """
    s = stem.lower()
    if re.search(r"extens", s):
        return "Extension"
    if re.search(r"flex", s):
        return "Flexion"
    return "Neutral"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, type=str)
    p.add_argument("--gt_dir",   required=True, type=str)
    p.add_argument("--img_dir",  required=True, type=str,
                   help="folder of test images (DICOM preferred for PixelSpacing)")
    p.add_argument("--out_csv",  required=True, type=str)
    p.add_argument("--mm_per_px", default=None, type=float,
                   help="if given, overrides DICOM PixelSpacing for all images")
    p.add_argument("--fallback_mm_per_px", default=None, type=float,
                   help="used when an image has no DICOM PixelSpacing")
    p.add_argument("--pixel_only", action="store_true",
                   help="skip mm conversion entirely (multi-center data with no "
                        "unified spacing); err_mm column will equal err_px")
    p.add_argument("--img_exts", default=".dcm,.png,.jpg,.jpeg,.bmp,.tif,.tiff", type=str)
    return p.parse_args()


def read_yolo_pose_txt(path: Path) -> Optional[Tuple[Tuple[float, float, float, float],
                                                     List[Tuple[float, float, int]]]]:
    """Return ((xc,yc,w,h), [(kpx,kpy,kpv), ...]) normalized. None if empty / no detection."""
    txt = path.read_text().strip()
    if not txt:
        return None
    # take first non-empty line (highest confidence by convention from infer_yolo.py)
    line = next((ln for ln in txt.splitlines() if ln.strip()), "")
    if not line:
        return None
    toks = line.split()
    if len(toks) < 5 + 35 * 3:
        raise ValueError(f"{path}: expected >= {5+35*3} tokens, got {len(toks)}")
    _cls = int(float(toks[0]))
    xc, yc, w, h = map(float, toks[1:5])
    kps: List[Tuple[float, float, int]] = []
    for i in range(35):
        kx = float(toks[5 + i*3 + 0])
        ky = float(toks[5 + i*3 + 1])
        kv = int(float(toks[5 + i*3 + 2]))
        kps.append((kx, ky, kv))
    return (xc, yc, w, h), kps


def find_image(img_dir: Path, stem: str, exts: Tuple[str, ...]) -> Optional[Path]:
    for ext in exts:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # also try recursive
    hits = list(img_dir.rglob(f"{stem}.*"))
    hits = [h for h in hits if h.suffix.lower() in exts]
    return hits[0] if hits else None


def read_image_size_and_spacing(img_path: Path, need_spacing: bool
                                ) -> Tuple[int, int, Optional[float]]:
    """Return (H, W, mm_per_px_or_None). If need_spacing=False, skip DICOM
    header parsing; use the fastest path to get dimensions only."""
    if img_path.suffix.lower() == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(img_path), stop_before_pixels=not need_spacing)
        if need_spacing:
            arr = ds.pixel_array
            H, W = arr.shape[:2]
            spacing = None
            for tag in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
                v = getattr(ds, tag, None)
                if v is not None:
                    try:
                        spacing = float(v[0]); break
                    except Exception:
                        pass
            return H, W, spacing
        H = int(getattr(ds, "Rows"))
        W = int(getattr(ds, "Columns"))
        return H, W, None
    # non-DICOM: use PIL (opens header only, ~1000x faster than cv2.imread)
    from PIL import Image
    with Image.open(str(img_path)) as im:
        W, H = im.size
    return H, W, None


def main() -> None:
    a = parse_args()
    pred_dir = Path(a.pred_dir)
    gt_dir   = Path(a.gt_dir)
    img_dir  = Path(a.img_dir)
    out_csv  = Path(a.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    img_exts = tuple(e.strip().lower() for e in a.img_exts.split(","))

    pred_files = {p.stem: p for p in pred_dir.glob("*.txt")}
    gt_files   = {p.stem: p for p in gt_dir.glob("*.txt")}
    common = sorted(set(pred_files) & set(gt_files))
    only_pred = sorted(set(pred_files) - set(gt_files))
    only_gt   = sorted(set(gt_files) - set(pred_files))
    print(f"[compute_errors] {len(common)} paired, {len(only_pred)} pred-only, {len(only_gt)} gt-only")
    if only_pred[:3]:
        print(f"  pred-only sample: {only_pred[:3]}")
    if only_gt[:3]:
        print(f"  gt-only sample: {only_gt[:3]}")

    records: List[tuple] = []
    missing_spacing: List[str] = []
    skipped: List[str] = []

    for stem in common:
        try:
            pred = read_yolo_pose_txt(pred_files[stem])
            gt   = read_yolo_pose_txt(gt_files[stem])
        except Exception as e:
            print(f"[WARN] parse fail {stem}: {e}")
            skipped.append(stem); continue
        if pred is None or gt is None:
            skipped.append(stem); continue

        img_path = find_image(img_dir, stem, img_exts)
        if img_path is None:
            print(f"[WARN] image not found for {stem}")
            skipped.append(stem); continue
        H, W, mm_per_px = read_image_size_and_spacing(img_path, need_spacing=not a.pixel_only)

        if a.pixel_only:
            mm_per_px = 1.0                             # err_mm will equal err_px
        elif a.mm_per_px is not None:
            mm_per_px = a.mm_per_px
        elif mm_per_px is None:
            if a.fallback_mm_per_px is not None:
                mm_per_px = a.fallback_mm_per_px
            else:
                missing_spacing.append(stem)
                continue

        _, pred_kps = pred
        _, gt_kps   = gt
        for i, (vert, kp) in enumerate(FLAT_KP):
            px, py, pv = pred_kps[i]
            gx, gy, gv = gt_kps[i]
            if gv == 0 or (px == 0 and py == 0):
                continue                                # missing GT or absent pred
            # YOLO-pose txt uses normalized [0,1] -> convert to pixel coords
            pxx = px * W; pyy = py * H
            gxx = gx * W; gyy = gy * H
            err_px = math.hypot(pxx - gxx, pyy - gyy)
            err_mm = err_px * mm_per_px
            records.append((
                stem, infer_view_from_name(stem), vert, kp,
                err_px, err_mm, mm_per_px, pxx, pyy, gxx, gyy,
            ))

    if missing_spacing:
        print(f"[ERROR] {len(missing_spacing)} images had NO PixelSpacing and no "
              f"--fallback_mm_per_px provided. Skipped. First 5: {missing_spacing[:5]}")
    print(f"[compute_errors] wrote {len(records)} keypoint records from "
          f"{len(common) - len(skipped) - len(missing_spacing)} images -> {out_csv}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "view", "vertebra", "keypoint",
                    "err_px", "err_mm", "mm_per_px",
                    "pred_x", "pred_y", "gt_x", "gt_y"])
        for (stem, view, vert, kp, ep, em, mpp, px, py, gx, gy) in records:
            w.writerow([stem, view, vert, kp,
                        f"{ep:.4f}", f"{em:.4f}", f"{mpp:.6f}",
                        f"{px:.2f}", f"{py:.2f}", f"{gx:.2f}", f"{gy:.2f}"])

    from collections import defaultdict, Counter
    import statistics as st
    view_counts = Counter(r[1] for r in records)
    print(f"[view mix] {dict(view_counts)}")
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in records:
        groups[(r[2], r[3])].append(r[5])
    unit = "px" if a.pixel_only else "mm"
    # in pixel_only mode err_px and err_mm are equal; print the meaningful one
    col_idx = 4 if a.pixel_only else 5
    groups_px_for_print: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in records:
        groups_px_for_print[(r[2], r[3])].append(r[col_idx])
    print(f"\n[summary pooled] mean ± sd / median [IQR]  of err_{unit}")
    for (v, k), vals in sorted(groups_px_for_print.items()):
        if not vals: continue
        m, sd = st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)
        vs = sorted(vals)
        q1 = vs[len(vs)//4]; med = vs[len(vs)//2]; q3 = vs[(3*len(vs))//4]
        print(f"  {v}-{k:2s}  n={len(vals):4d}  mean={m:.2f}±{sd:.2f}  med={med:.2f} [{q1:.2f},{q3:.2f}]")


if __name__ == "__main__":
    main()
