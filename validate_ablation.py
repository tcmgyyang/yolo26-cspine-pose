"""
Batch model validation for the component-ablation experiments.

Validates each ablation configuration's `best.pt` on the test split and emits
the per-model and cross-model comparison tables used in the paper.
"""

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from prettytable import PrettyTable
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================
BASE_DIR = 'runs/pose/spine_extreme/ablation'
DATA_YAML = 'dataset/data.yaml'
OUTPUT_DIR = 'runs/pose/spine_extreme/ablation/validation_results'

# Ablation lineup: (run-dir name, display label, DPCA, DySample, MS-OKS)
ABLATION_MODELS = [
    ('v1_1280px_yolo26s_pose_Pose30',         'Baseline',          '-', '-', '-'),
    ('ablation_DPCA_only',                     '+DPCA',             'Y', '-', '-'),
    ('ablation_DySample_only',                 '+DySample',         '-', 'Y', '-'),
    ('ablation_MSOKS_only',                    '+MS-OKS',           '-', '-', 'Y'),
    ('yolo26s_yolo26s-pose-dpca-AC_nosig',     '+DPCA+DySample',    'Y', 'Y', '-'),
    ('ablation_DPCA_MSOKS',                    '+DPCA+MS-OKS',      'Y', '-', 'Y'),
    ('ablation_DySample_MSOKS',                '+DySample+MS-OKS',  '-', 'Y', 'Y'),
    ('DPCA_AC_sigma',                          'Full (Ours)',       'Y', 'Y', 'Y'),
]

# ============================================================
# Helpers
# ============================================================
def get_weight_size(path):
    """Return the weight file size in MB as a formatted string."""
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def validate_single_model(model_name, label, base_dir, data_yaml, output_dir):
    """Validate one ablation run directory and return its metrics dictionary."""
    model_path = os.path.join(base_dir, model_name, 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"[skip] weight file not found: {model_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Validating: {label} ({model_name})")
    print(f"Weight:     {model_path}")
    print(f"{'='*60}")

    try:
        model = YOLO(model_path)
        result = model.val(
            data=data_yaml,
            split='test',
            imgsz=1280,
            batch=16,
            save_json=True,
            project=output_dir,
            name=model_name,
        )

        if model.task != 'pose':
            print(f"[warn] {model_name} is not a pose task, skipping.")
            return None

        # Timing
        preprocess_time = result.speed['preprocess']
        inference_time = result.speed['inference']
        postprocess_time = result.speed['postprocess']
        all_time = preprocess_time + inference_time + postprocess_time

        # Model parameters / FLOPs
        try:
            n_l, n_p, n_g, flops = model.model.info(verbose=False)
        except:
            try:
                n_p = sum(x.numel() for x in model.parameters())
                from ultralytics.utils.torch_utils import get_flops
                flops = get_flops(model.model, imgsz=1280)
            except:
                n_p = 0
                flops = 0.0

        # ================= Assemble results dict =================
        results_dict = {
            'model_name': model_name,
            'model_label': label,
            'gflops': flops,
            'parameters': n_p,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'fps_all': 1000 / all_time,
            'fps_inference': 1000 / inference_time,
            'model_size': get_weight_size(model_path),
        }

        # Box metrics
        try:
            results_dict['box_precision'] = result.results_dict.get('metrics/precision(B)', 0)
            results_dict['box_recall'] = result.results_dict.get('metrics/recall(B)', 0)
            results_dict['box_map50'] = result.results_dict.get('metrics/mAP50(B)', 0)
            results_dict['box_map'] = result.results_dict.get('metrics/mAP50-95(B)', 0)
            results_dict['box_map75'] = np.mean(result.box.all_ap[:, 5]) if result.box.all_ap is not None else 0
            b_p, b_r = results_dict['box_precision'], results_dict['box_recall']
            results_dict['box_f1'] = 2 * (b_p * b_r) / (b_p + b_r + 1e-16)
        except:
            for k in ['box_precision', 'box_recall', 'box_map50', 'box_map', 'box_map75', 'box_f1']:
                results_dict[k] = 0

        # Pose metrics
        try:
            results_dict['pose_precision'] = result.results_dict.get('metrics/precision(P)', 0)
            results_dict['pose_recall'] = result.results_dict.get('metrics/recall(P)', 0)
            results_dict['pose_map50'] = result.results_dict.get('metrics/mAP50(P)', 0)
            results_dict['pose_map'] = result.results_dict.get('metrics/mAP50-95(P)', 0)
            results_dict['pose_map75'] = np.mean(result.pose.all_ap[:, 5]) if result.pose.all_ap is not None else 0
            p_p, p_r = results_dict['pose_precision'], results_dict['pose_recall']
            results_dict['pose_f1'] = 2 * (p_p * p_r) / (p_p + p_r + 1e-16)
        except:
            for k in ['pose_precision', 'pose_recall', 'pose_map50', 'pose_map', 'pose_map75', 'pose_f1']:
                results_dict[k] = 0

        # Per-model report
        generate_single_report(results_dict, result.save_dir)

        return results_dict

    except Exception as e:
        print(f"[fail] validation failed for {model_name}")
        print(f"       error: {str(e)}")
        return None


def generate_single_report(results, save_dir):
    """Emit a single-model report (printed + paper_data.txt)."""
    model_info_table = PrettyTable()
    model_info_table.title = f"Model Info - {results['model_label']}"
    model_info_table.field_names = [
        "GFLOPs", "Parameters",
        "Preprocess/img", "Inference/img", "Postprocess/img",
        "FPS (end-to-end)", "FPS (inference only)", "Model Size",
    ]
    model_info_table.add_row([
        f"{results['gflops']:.1f}",
        f"{results['parameters']:,}",
        f"{results['preprocess_time'] / 1000:.4f}s",
        f"{results['inference_time'] / 1000:.4f}s",
        f"{results['postprocess_time'] / 1000:.4f}s",
        f"{results['fps_all']:.2f}",
        f"{results['fps_inference']:.2f}",
        f"{results['model_size']}MB"
    ])

    model_metrice_table = PrettyTable()
    model_metrice_table.title = "Model Metrics (Box & Pose)"
    model_metrice_table.field_names = ["Task", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
    model_metrice_table.add_row([
        "Box",
        f"{results['box_precision']:.4f}",
        f"{results['box_recall']:.4f}",
        f"{results['box_f1']:.4f}",
        f"{results['box_map50']:.4f}",
        f"{results['box_map75']:.4f}",
        f"{results['box_map']:.4f}"
    ])
    model_metrice_table.add_row([
        "Pose",
        f"{results['pose_precision']:.4f}",
        f"{results['pose_recall']:.4f}",
        f"{results['pose_f1']:.4f}",
        f"{results['pose_map50']:.4f}",
        f"{results['pose_map75']:.4f}",
        f"{results['pose_map']:.4f}"
    ])

    print(model_info_table)
    print(model_metrice_table)

    save_path = Path(save_dir) / 'paper_data.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(str(model_info_table))
        f.write('\n\n')
        f.write(str(model_metrice_table))
    print(f"[OK] saved to: {save_path}")


def generate_comparison_report(all_results, components_map, output_dir):
    """Emit the cross-model ablation comparison report."""
    if not all_results:
        print("No valid validation results — cannot build comparison report.")
        return

    os.makedirs(output_dir, exist_ok=True)
    baseline_pose_map = all_results[0]['pose_map'] if all_results else 0

    # ================= Table 1: ablation comparison (manuscript-ready) =================
    ablation_table = PrettyTable()
    ablation_table.title = "Component Ablation Study"
    ablation_table.field_names = [
        "DPCA", "DySample", "MS-OKS", "Params(M)", "GFLOPs",
        "Box mAP50", "Box mAP50-95", "Pose mAP50", "Pose mAP50-95", "Δ"
    ]

    for r in all_results:
        comp = components_map[r['model_name']]
        is_baseline = comp[0] == '—' and comp[1] == '—' and comp[2] == '—'
        delta_val = r['pose_map'] - baseline_pose_map
        delta = '—' if is_baseline else f"+{delta_val:.4f}"

        ablation_table.add_row([
            comp[0], comp[1], comp[2],
            f"{r['parameters'] / 1e6:.2f}",
            f"{r['gflops']:.1f}",
            f"{r['box_map50']:.4f}",
            f"{r['box_map']:.4f}",
            f"{r['pose_map50']:.4f}",
            f"{r['pose_map']:.4f}",
            delta,
        ])

    # ================= Table 2: basic info comparison =================
    info_table = PrettyTable()
    info_table.title = "All Models - Basic Info Comparison"
    info_table.field_names = ["Label", "Model", "GFLOPs", "Params(M)", "FPS (end-to-end)", "FPS (inference)", "Size(MB)"]

    for r in all_results:
        info_table.add_row([
            r['model_label'],
            r['model_name'],
            f"{r['gflops']:.1f}",
            f"{r['parameters'] / 1e6:.2f}",
            f"{r['fps_all']:.2f}",
            f"{r['fps_inference']:.2f}",
            r['model_size']
        ])

    # ================= Table 3: Box detection metrics =================
    box_table = PrettyTable()
    box_table.title = "All Models - Box Detection Metrics"
    box_table.field_names = ["Label", "Precision", "Recall", "F1", "mAP50", "mAP75", "mAP50-95"]

    for r in all_results:
        box_table.add_row([
            r['model_label'],
            f"{r['box_precision']:.4f}",
            f"{r['box_recall']:.4f}",
            f"{r['box_f1']:.4f}",
            f"{r['box_map50']:.4f}",
            f"{r['box_map75']:.4f}",
            f"{r['box_map']:.4f}"
        ])

    # ================= Table 4: Pose keypoint metrics =================
    pose_table = PrettyTable()
    pose_table.title = "All Models - Pose Estimation Metrics"
    pose_table.field_names = ["Label", "Precision", "Recall", "F1", "mAP50", "mAP75", "mAP50-95"]

    for r in all_results:
        pose_table.add_row([
            r['model_label'],
            f"{r['pose_precision']:.4f}",
            f"{r['pose_recall']:.4f}",
            f"{r['pose_f1']:.4f}",
            f"{r['pose_map50']:.4f}",
            f"{r['pose_map75']:.4f}",
            f"{r['pose_map']:.4f}"
        ])

    # Print
    print("\n" + "=" * 80)
    print("Ablation — cross-model comparison report")
    print("=" * 80)
    print(ablation_table)
    print(info_table)
    print(box_table)
    print(pose_table)

    # Save TXT
    report_path = os.path.join(output_dir, 'ablation_comparison.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Ablation comparison report — generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(ablation_table))
        f.write('\n\n')
        f.write(str(info_table))
        f.write('\n\n')
        f.write(str(box_table))
        f.write('\n\n')
        f.write(str(pose_table))
    print(f"\n[OK] comparison report saved to: {report_path}")

    # ================= CSV =================
    csv_path = os.path.join(output_dir, 'ablation_comparison.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Label,DPCA,DySample,MS-OKS,GFLOPs,Params(M),FPS,Size(MB),")
        f.write("Box_P,Box_R,Box_F1,Box_mAP50,Box_mAP75,Box_mAP50-95,")
        f.write("Pose_P,Pose_R,Pose_F1,Pose_mAP50,Pose_mAP75,Pose_mAP50-95\n")

        for r in all_results:
            comp = components_map[r['model_name']]
            f.write(f"{r['model_label']},{comp[0]},{comp[1]},{comp[2]},")
            f.write(f"{r['gflops']:.1f},{r['parameters']/1e6:.2f},")
            f.write(f"{r['fps_inference']:.2f},{r['model_size']},")
            f.write(f"{r['box_precision']:.4f},{r['box_recall']:.4f},{r['box_f1']:.4f},")
            f.write(f"{r['box_map50']:.4f},{r['box_map75']:.4f},{r['box_map']:.4f},")
            f.write(f"{r['pose_precision']:.4f},{r['pose_recall']:.4f},{r['pose_f1']:.4f},")
            f.write(f"{r['pose_map50']:.4f},{r['pose_map75']:.4f},{r['pose_map']:.4f}\n")

    print(f"[OK] CSV saved to: {csv_path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Batch validation — component ablation")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Build a {run-dir -> (DPCA, DySample, MS-OKS)} map
    components_map = {}
    for folder, label, dpca, ds, msoks in ABLATION_MODELS:
        components_map[folder] = (dpca, ds, msoks)

    # Check which weight files exist
    valid_models = []
    for folder, label, dpca, ds, msoks in ABLATION_MODELS:
        model_path = os.path.join(BASE_DIR, folder, 'weights', 'best.pt')
        if os.path.exists(model_path):
            valid_models.append((folder, label))
            print(f"  [ok]   [{dpca} {ds} {msoks}] {label}: {folder}")
        else:
            print(f"  [miss] [{dpca} {ds} {msoks}] {label}: not found {model_path}")

    print(f"\nValid models: {len(valid_models)}/{len(ABLATION_MODELS)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Batch validation
    all_results = []
    for i, (folder, label) in enumerate(valid_models, 1):
        print(f"\n[{i}/{len(valid_models)}] validating: {label}")
        result = validate_single_model(folder, label, BASE_DIR, DATA_YAML, OUTPUT_DIR)
        if result:
            all_results.append(result)

    # Comparison report
    generate_comparison_report(all_results, components_map, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Ablation validation complete.")
    print(f"Successful: {len(all_results)}/{len(valid_models)}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
