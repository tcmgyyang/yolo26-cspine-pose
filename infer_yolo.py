"""
Run YOLO-pose inference on a folder of test images and export predictions in
standard YOLO-pose txt format (one file per image, same basename as image).

Line format (one line per detection; normalized to image W,H):
    cls xc yc w h  (kpx kpy kpv) * N_KPTS

Usage (on server):
    python infer_yolo.py \
        --weights /path/to/best.pt \
        --images  /path/to/test/images \
        --out     /path/to/pred_txt \
        --conf    0.25 \
        --imgsz   1280 \
        --device  0

If there is more than one detected object per image, only the highest-confidence
detection is written (the test set here has exactly one spine per image).
"""
from __future__ import annotations
import argparse, os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=str, help="YOLO-pose .pt checkpoint")
    p.add_argument("--images",  required=True, type=str, help="folder of test images")
    p.add_argument("--out",     required=True, type=str, help="output folder for YOLO-pose txt")
    p.add_argument("--conf",    default=0.25, type=float)
    p.add_argument("--iou",     default=0.7,  type=float)
    p.add_argument("--imgsz",   default=1280, type=int)
    p.add_argument("--device",  default="0",  type=str)
    p.add_argument("--exts",    default=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.dcm", type=str)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    # Prefer the canonical top-level import; fall back to a deep path that
    # bypasses ultralytics.models' lazy SAM/torchvision/onnx loader if that
    # chain is broken by downstream dependency drift.
    try:
        from ultralytics import YOLO
    except ImportError:
        from ultralytics.models.yolo.model import YOLO
    import numpy as np, cv2

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)

    exts = tuple(e.strip().lower() for e in a.exts.split(","))
    imgs = sorted([p for p in Path(a.images).rglob("*") if p.suffix.lower() in exts])
    print(f"[infer_yolo] {len(imgs)} images found under {a.images}")

    for i, img_path in enumerate(imgs, 1):
        # Preload image to a BGR numpy array ourselves, bypassing ultralytics'
        # patched imread (which can break on custom patches.py / opencv drift).
        if img_path.suffix.lower() == ".dcm":
            import pydicom
            ds = pydicom.dcmread(str(img_path))
            arr = ds.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.ptp() + 1e-6) * 255.0
            img = arr.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            buf = np.fromfile(str(img_path), dtype=np.uint8)
            if buf.size == 0:
                print(f"[WARN] empty/unreadable file: {img_path}")
                (out_dir / (img_path.stem + ".txt")).write_text("")
                continue
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] cv2.imdecode failed: {img_path}")
                (out_dir / (img_path.stem + ".txt")).write_text("")
                continue

        res = model.predict(img, conf=a.conf, iou=a.iou, imgsz=a.imgsz,
                            device=a.device, verbose=False)[0]

        H, W = res.orig_shape
        if res.boxes is None or len(res.boxes) == 0:
            print(f"[WARN] no detection: {img_path.name}")
            # still write empty file so pairing is explicit downstream
            (out_dir / (img_path.stem + ".txt")).write_text("")
            continue

        # keep the highest-confidence detection
        k = int(res.boxes.conf.argmax().item())
        xywh = res.boxes.xywhn[k].cpu().numpy()            # (4,)  normalized
        cls  = int(res.boxes.cls[k].item())
        # keypoints: Ultralytics gives .xy (absolute) and .xyn (normalized). Visibility in .conf.
        kxyn = res.keypoints.xyn[k].cpu().numpy()          # (N, 2)
        kconf = res.keypoints.conf
        if kconf is not None:
            v = kconf[k].cpu().numpy()                     # (N,)
            # YOLO-pose txt convention: visibility flag 2=visible, 1=occluded, 0=absent.
            # We use 2 if conf > 0.5, 1 otherwise (non-zero kept for parser compat).
            v = (v > 0.5).astype(np.int32) + 1
        else:
            v = np.full((kxyn.shape[0],), 2, dtype=np.int32)

        toks = [str(cls), f"{xywh[0]:.6f}", f"{xywh[1]:.6f}", f"{xywh[2]:.6f}", f"{xywh[3]:.6f}"]
        for (kx, ky), vv in zip(kxyn, v):
            toks.extend([f"{kx:.6f}", f"{ky:.6f}", str(int(vv))])
        (out_dir / (img_path.stem + ".txt")).write_text(" ".join(toks) + "\n")

        if i % 50 == 0 or i == len(imgs):
            print(f"[infer_yolo] {i}/{len(imgs)}")

    print(f"[infer_yolo] DONE. txt written to {out_dir}")


if __name__ == "__main__":
    main()
