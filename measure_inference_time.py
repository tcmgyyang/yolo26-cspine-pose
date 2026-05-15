"""
Measure end-to-end inference time on a random sample of test images.

Outputs a CSV with the same column layout as `manual_annotation_times.csv`
so the two can be concatenated downstream for figure rendering.

Usage:  python measure_inference_time.py
"""

import warnings
warnings.filterwarnings('ignore')
import os
import random
import time
import numpy as np
import csv
from ultralytics import YOLO

# ============================================================
# Configuration (edit as needed)
# ============================================================
MODEL_PATH = 'runs/pose/spine_extreme/DPCA/DPCA_AC_sigma/weights/best.pt'
DATA_DIR   = 'dataset/images/test'   # directory of test images
N_SAMPLES  = 30
IMGSZ      = 1280
OUTPUT_CSV = 'runs/pose/spine_extreme/baseline/model_inference_times_30samples.csv'

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    # Enumerate test images
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    all_images = sorted([
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
        if f.lower().endswith(img_extensions)
    ])

    print(f"Test set contains {len(all_images)} images")

    # Random sub-sample
    random.seed(42)
    sampled_images = random.sample(all_images, min(N_SAMPLES, len(all_images)))
    print(f"Sampled {len(sampled_images)} images for timing\n")

    # Load the model
    model = YOLO(MODEL_PATH)

    # Warm-up (5 passes, not counted)
    print("Warming up...")
    for _ in range(5):
        _ = model.predict(sampled_images[0], imgsz=IMGSZ, verbose=False)
    print("Warm-up done.\n")

    # Per-image inference and timing
    results_list = []

    for i, img_path in enumerate(sampled_images):
        img_name = os.path.basename(img_path)

        # End-to-end wall-clock time
        start = time.perf_counter()
        result = model.predict(img_path, imgsz=IMGSZ, verbose=False)
        end = time.perf_counter()

        total_ms = (end - start) * 1000
        speed = result[0].speed

        results_list.append({
            'Image_ID': f'img_{i+1:03d}',
            'filename': img_name,
            'preprocess_ms': speed['preprocess'],
            'inference_ms': speed['inference'],
            'postprocess_ms': speed['postprocess'],
            'total_ms': total_ms,
        })

        print(f"[{i+1:2d}/{len(sampled_images)}] {img_name}: "
              f"pre={speed['preprocess']:.2f}ms  infer={speed['inference']:.2f}ms  "
              f"post={speed['postprocess']:.2f}ms  e2e={total_ms:.2f}ms")

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Image_ID', 'filename', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms'
        ])
        writer.writeheader()
        writer.writerows(results_list)

    # Summary stats
    total_times = [r['total_ms'] for r in results_list]
    inference_times = [r['inference_ms'] for r in results_list]

    print(f"\n{'='*60}")
    print(f"Summary ({len(results_list)} images)")
    print(f"{'='*60}")
    print(f"End-to-end total:  mean={np.mean(total_times):.2f}ms  std={np.std(total_times):.2f}ms")
    print(f"Inference only:    mean={np.mean(inference_times):.2f}ms  std={np.std(inference_times):.2f}ms")
    print(f"\nSaved {len(results_list)} timing records to: {OUTPUT_CSV}")
