"""
YOLO26-Pose ablation visualisation (CAM heatmap + keypoints + composite figure).

Usage:
  python ablation_visualization.py --image test.jpg
  python ablation_visualization.py --image test.jpg --layer 10
"""

import os, sys, cv2, torch, argparse
import numpy as np
from pathlib import Path

# ==================== Configuration ====================
BASE_DIR = "runs/pose/spine_extreme/ablation"

MODELS = [
    ("Ours",                  os.path.join(BASE_DIR, "DPCA_AC_sigma/weights/best.pt")),
    ("w/o DySample & MSOKS",  os.path.join(BASE_DIR, "ablation_DPCA_only/weights/best.pt")),
    ("w/o DPCA & MSOKS",      os.path.join(BASE_DIR, "ablation_DySample_only/weights/best.pt")),
    ("w/o DPCA & DySample",   os.path.join(BASE_DIR, "ablation_MSOKS_only/weights/best.pt")),
]

OUTPUT_DIR = os.path.join(BASE_DIR, "validation_results_ablation")
CONF = 0.25
IMGSZ = 1280
SKELETON = None  # None = use the auto adjacency edges


# ==================== Heatmap generation (minimal) ====================

def get_heatmap(model, img_path, results, layer_idx=None, imgsz=1280):
    """
    Generate a class-agnostic activation heatmap. The target region is
    guaranteed to render in warm colour.

    How it works:
      1. Hook the target layer and capture its feature map.
      2. Sum over channels to produce a raw CAM.
      3. Use the detection bbox to decide whether to flip the polarity.
      4. Overlay the CAM on the original image.
    """
    torch_model = model.model
    torch_model.eval()
    device = next(torch_model.parameters()).device

    # --- Locate the target layer ---
    if layer_idx is not None:
        target_layer = torch_model.model[layer_idx]
    else:
        target_layer = None
        for i in range(min(12, len(torch_model.model))):
            if type(torch_model.model[i]).__name__ in ["SPPF", "SPP", "C2f", "C3"]:
                target_layer = torch_model.model[i]
                break
        if target_layer is None:
            target_layer = torch_model.model[min(9, len(torch_model.model) - 2)]

    # --- hook ---
    stored = {}

    def hook_fn(m, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            stored["acts"] = out.detach()
        elif isinstance(out, (list, tuple)):
            for o in out:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    stored["acts"] = o.detach()
                    break

    handle = target_layer.register_forward_hook(hook_fn)

    # --- Forward pass ---
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]
    img_r = cv2.resize(img, (imgsz, imgsz))
    t = torch.from_numpy(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0) / 255.0
    with torch.no_grad():
        torch_model(t.to(device))
    handle.remove()

    if "acts" not in stored:
        print("  [WARN] no activation captured; returning empty heatmap")
        return np.zeros((h0, w0), dtype=np.float32)

    acts = stored["acts"].squeeze(0).cpu().numpy()  # [C, fH, fW]

    # --- Compute CAM: sum over channels ---
    cam = acts.sum(axis=0)  # [fH, fW]

    # Normalise to [0, 1]
    cmin, cmax = cam.min(), cam.max()
    if cmax > cmin:
        cam = (cam - cmin) / (cmax - cmin)
    else:
        cam = np.zeros_like(cam)

    # --- Use the detection bbox to fix the polarity ---
    # Resize to the original image size before comparing.
    cam_full = cv2.resize(cam, (w0, h0))

    bboxes = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            xy = box.xyxy[0].cpu().numpy().astype(int)
            bboxes.append((max(0,xy[0]), max(0,xy[1]), min(w0,xy[2]), min(h0,xy[3])))

    if len(bboxes) > 0:
        mask = np.zeros((h0, w0), dtype=bool)
        for x1, y1, x2, y2 in bboxes:
            mask[y1:y2, x1:x2] = True

        if mask.sum() > 0 and (~mask).sum() > 0:
            val_inside = cam_full[mask].mean()
            val_outside = cam_full[~mask].mean()
            print(f"  [CAM] mean inside={val_inside:.3f}, mean outside={val_outside:.3f}", end="")
            if val_inside < val_outside:
                cam_full = 1.0 - cam_full
                print(" -> polarity flipped")
            else:
                print(" -> polarity OK")

    return cam_full


def overlay_heatmap(img_bgr, cam, alpha=0.5):
    """Overlay a CAM [H, W] onto the original image; returns BGR."""
    h, w = img_bgr.shape[:2]
    cam_r = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


# ==================== Keypoint rendering ====================

def draw_results(img, results):
    """Draw bbox + keypoints onto a copy of `img`."""
    out = img.copy()
    if not results or len(results) == 0:
        return out
    r = results[0]

    # bbox
    if r.boxes is not None:
        for box in r.boxes:
            xy = box.xyxy[0].cpu().numpy().astype(int)
            c = box.conf[0].item()
            cv2.rectangle(out, (xy[0],xy[1]), (xy[2],xy[3]), (0,255,255), 2)
            cv2.putText(out, f"{c:.2f}", (xy[0], xy[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

    # keypoints (dots only — no skeleton edges)
    if r.keypoints is not None:
        for kd in r.keypoints:
            kpts = kd.data[0].cpu().numpy()
            has_c = kpts.shape[1] >= 3

            for kp in kpts:
                c = kp[2] if has_c else 1
                if c > 0.3:
                    cv2.circle(out, (int(kp[0]),int(kp[1])), 4, (0,255,0), -1, cv2.LINE_AA)
                    cv2.circle(out, (int(kp[0]),int(kp[1])), 5, (255,255,255), 1, cv2.LINE_AA)

    return out


def crop_roi(img, results, pad=0.15):
    """Crop the detection ROI."""
    h, w = img.shape[:2]
    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = boxes[:,0].min(), boxes[:,1].min(), boxes[:,2].max(), boxes[:,3].max()
        bw, bh = x2-x1, y2-y1
        x1, y1 = max(0, int(x1-bw*pad)), max(0, int(y1-bh*pad))
        x2, y2 = min(w, int(x2+bw*pad)), min(h, int(y2+bh*pad))
    else:
        cx, cy, s = w//2, h//2, min(w,h)//2
        x1, y1, x2, y2 = max(0,cx-s), max(0,cy-s), min(w,cx+s), min(h,cy+s)
    return img[y1:y2, x1:x2].copy()


# ==================== Panels & layout ====================

def letterbox(img, target_w, target_h):
    """Aspect-preserving resize with black-bar padding — never distorts."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - nh) // 2
    x_off = (target_w - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas


def add_label(img, text, position="bottom"):
    """Overlay text on `img` with a semi-transparent dark background."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    h, w = out.shape[:2]

    if position == "bottom":
        lx, ly = (w - tw) // 2, h - 10
    else:
        lx, ly = (w - tw) // 2, th + 10

    ov = out.copy()
    cv2.rectangle(ov, (lx - 4, ly - th - 4), (lx + tw + 4, ly + baseline + 4), (0,0,0), -1)
    out = cv2.addWeighted(ov, 0.6, out, 0.4, 0)
    cv2.putText(out, text, (lx, ly), font, scale, (0,255,255), thickness, cv2.LINE_AA)
    return out


def build_comparison_figure(img_orig, kpt_crops, heat_crops, labels, gap=4):
    """
    Composite-figure layout:

        +-----------+--------+--------+--------+--------+
        |           |  Ours  | w/o A  | w/o B  | w/o C  |
        | original  | (kpts) | (kpts) | (kpts) | (kpts) |
        |           +--------+--------+--------+--------+
        |           |  Ours  | w/o A  | w/o B  | w/o C  |
        |           |  (cam) |  (cam) |  (cam) |  (cam) |
        +-----------+--------+--------+--------+--------+

    Args:
        img_orig:   the original X-ray (BGR)
        kpt_crops:  4 keypoint-annotated ROI crops (list of BGR)
        heat_crops: 4 CAM-overlay ROI crops      (list of BGR)
        labels:     4 column labels              (list of str)
    """
    n = len(labels)  # 4

    # Right-hand cell size (driven by the first kpt crop's aspect ratio)
    ref = kpt_crops[0]
    ref_ratio = ref.shape[0] / ref.shape[1]  # h/w

    # Right side: n columns x 2 rows
    cell_w = 280
    cell_h = int(cell_w * ref_ratio)
    cell_h = max(cell_h, 200)  # minimum cell height

    # Left-hand original image fills the same total height as the right side
    right_total_h = cell_h * 2 + gap
    left_w = int(right_total_h / (img_orig.shape[0] / img_orig.shape[1]))
    left_w = max(left_w, 300)

    # Final canvas
    right_total_w = cell_w * n + gap * (n - 1)
    total_w = left_w + gap + right_total_w
    total_h = right_total_h

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # === Left side: original ===
    canvas[:total_h, :left_w] = letterbox(img_orig, left_w, total_h)

    # === Right side, top row: keypoint crops ===
    for i in range(n):
        x = left_w + gap + i * (cell_w + gap)
        y = 0
        cell = letterbox(kpt_crops[i], cell_w, cell_h)
        cell = add_label(cell, labels[i], position="bottom")
        canvas[y:y+cell_h, x:x+cell_w] = cell

    # === Right side, bottom row: heatmap crops ===
    for i in range(n):
        x = left_w + gap + i * (cell_w + gap)
        y = cell_h + gap
        cell = letterbox(heat_crops[i], cell_w, cell_h)
        cell = add_label(cell, labels[i], position="bottom")
        canvas[y:y+cell_h, x:x+cell_w] = cell

    # Separator lines
    # Between left original and the right grid
    lx = left_w + gap // 2
    cv2.line(canvas, (lx, 0), (lx, total_h), (100,100,100), 1)
    # Between the top and bottom rows of the right grid
    ly = cell_h + gap // 2
    cv2.line(canvas, (left_w + gap, ly), (total_w, ly), (100,100,100), 1)

    return canvas


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="test image path")
    parser.add_argument("--layer", type=int, default=None, help="CAM target layer index")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"image not found: {args.image}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = args.output or os.path.join(OUTPUT_DIR, f"ablation_{Path(args.image).stem}.png")

    print("=" * 60)
    print("  Ablation visualisation")
    print("=" * 60)
    for label, wt in MODELS:
        ok = "ok " if os.path.exists(wt) else "miss"
        print(f"  [{ok}] {label}")
    print("=" * 60)

    from ultralytics import YOLO

    img_orig = cv2.imread(args.image)
    labels = []
    kpt_crops = []
    heat_crops = []

    for label, weight_path in MODELS:
        print(f"\n>>> {label}")
        if not os.path.exists(weight_path):
            print("  [SKIP] weight file not found")
            labels.append(label)
            placeholder = np.zeros((400, 300, 3), dtype=np.uint8)
            cv2.putText(placeholder, "N/A", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            kpt_crops.append(placeholder)
            heat_crops.append(placeholder)
            continue

        # Load model + run inference
        model = YOLO(weight_path)
        results = model.predict(source=args.image, conf=args.conf, imgsz=args.imgsz, verbose=False)

        # Draw keypoints + crop ROI
        img_det = draw_results(img_orig.copy(), results)
        img_crop = crop_roi(img_det, results)

        # Heatmap + crop
        print("  generating heatmap...")
        cam = get_heatmap(model, args.image, results, layer_idx=args.layer, imgsz=args.imgsz)
        img_heat_full = overlay_heatmap(img_orig, cam)
        img_heat_crop = crop_roi(img_heat_full, results)

        labels.append(label)
        kpt_crops.append(img_crop)
        heat_crops.append(img_heat_crop)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [OK] done")

    # Composite
    canvas = build_comparison_figure(img_orig, kpt_crops, heat_crops, labels)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    jpg = out_path.replace(".png", ".jpg")
    cv2.imwrite(jpg, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n{'='*60}")
    print(f"  [OK] {out_path}")
    print(f"  [OK] {jpg}")
    print(f"  size: {canvas.shape[1]}x{canvas.shape[0]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
