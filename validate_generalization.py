"""
Validate the cross-domain generalization experiments and emit manuscript tables.

Cross-domain configurations:
  - Clinical train  ->  public test
  - Public train    ->  clinical test
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
BASE_DIR = 'runs/pose/spine_extreme/generalization2'
OUTPUT_DIR = 'runs/pose/spine_extreme/generalization2/validation_results'

# Map each run directory to the test-set YAML it should be evaluated on
MODEL_DATA_MAP = {
    'generalization_clinical_train_public_test3': 'dataset/data2.yaml',
    'generalization_public_train_clinical_test': 'dataset/data3.yaml',
}

# ============================================================
# Helpers
# ============================================================
def get_weight_size(path):
    """Return the weight file size in MB as a formatted string."""
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def validate_single_model(model_name, base_dir, data_yaml, output_dir):
    """Validate one generalization run directory and return its metrics dictionary."""
    model_path = os.path.join(base_dir, model_name, 'weights', 'best.pt')

    print(f"\n{'='*60}")
    print(f"Validating: {model_name}")
    print(f"Weight:     {model_path}")
    print(f"Dataset:    {data_yaml}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"[skip] weight file not found: {model_path}")
        return None
    
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

        # Extract model info
        model_names = list(result.names.values())
        
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
            'data_yaml': data_yaml,
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
            results_dict['box_precision'] = 0
            results_dict['box_recall'] = 0
            results_dict['box_map50'] = 0
            results_dict['box_map'] = 0
            results_dict['box_map75'] = 0
            results_dict['box_f1'] = 0
        
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
            results_dict['pose_precision'] = 0
            results_dict['pose_recall'] = 0
            results_dict['pose_map50'] = 0
            results_dict['pose_map'] = 0
            results_dict['pose_map75'] = 0
            results_dict['pose_f1'] = 0
        
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
    model_info_table.title = f"Model Info - {results['model_name']}"
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
    
    # Save to file
    save_path = Path(save_dir) / 'paper_data.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {results['data_yaml']}\n\n")
        f.write(str(model_info_table))
        f.write('\n\n')
        f.write(str(model_metrice_table))
    print(f"[OK] saved to: {save_path}")


def generate_comparison_report(all_results, output_dir):
    """Emit the cross-model generalization comparison report."""
    if not all_results:
        print("No valid validation results — cannot build comparison report.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ================= Table 1: basic info comparison =================
    info_table = PrettyTable()
    info_table.title = "Generalization Experiment - Basic Info"
    info_table.field_names = ["Model", "GFLOPs", "Params(M)", "FPS (end-to-end)", "FPS (inference)", "Size(MB)"]
    
    for r in all_results:
        info_table.add_row([
            r['model_name'],
            f"{r['gflops']:.1f}",
            f"{r['parameters'] / 1e6:.2f}",
            f"{r['fps_all']:.2f}",
            f"{r['fps_inference']:.2f}",
            r['model_size']
        ])
    
    # ================= Table 2: Box detection metrics =================
    box_table = PrettyTable()
    box_table.title = "Generalization Experiment - Box Detection Metrics"
    box_table.field_names = ["Model", "Precision", "Recall", "F1", "mAP50", "mAP75", "mAP50-95"]
    
    for r in all_results:
        box_table.add_row([
            r['model_name'],
            f"{r['box_precision']:.4f}",
            f"{r['box_recall']:.4f}",
            f"{r['box_f1']:.4f}",
            f"{r['box_map50']:.4f}",
            f"{r['box_map75']:.4f}",
            f"{r['box_map']:.4f}"
        ])
    
    # ================= Table 3: Pose keypoint metrics =================
    pose_table = PrettyTable()
    pose_table.title = "Generalization Experiment - Pose Estimation Metrics"
    pose_table.field_names = ["Model", "Precision", "Recall", "F1", "mAP50", "mAP75", "mAP50-95"]
    
    for r in all_results:
        pose_table.add_row([
            r['model_name'],
            f"{r['pose_precision']:.4f}",
            f"{r['pose_recall']:.4f}",
            f"{r['pose_f1']:.4f}",
            f"{r['pose_map50']:.4f}",
            f"{r['pose_map75']:.4f}",
            f"{r['pose_map']:.4f}"
        ])
    
    # Print all tables
    print("\n" + "=" * 80)
    print("Generalization — cross-model comparison report")
    print("=" * 80)
    print(info_table)
    print(box_table)
    print(pose_table)

    # Save comparison report
    report_path = os.path.join(output_dir, 'generalization_comparison.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Generalization comparison report — generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Experimental design:\n")
        f.write("  1. Clinical train  ->  public test\n")
        f.write("  2. Public train    ->  clinical test\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(info_table))
        f.write('\n\n')
        f.write(str(box_table))
        f.write('\n\n')
        f.write(str(pose_table))

    print(f"\n[OK] comparison report saved to: {report_path}")

    # ================= CSV export for the manuscript =================
    csv_path = os.path.join(output_dir, 'generalization_comparison.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("Model,Dataset,GFLOPs,Params(M),FPS,Size(MB),")
        f.write("Box_P,Box_R,Box_F1,Box_mAP50,Box_mAP75,Box_mAP50-95,")
        f.write("Pose_P,Pose_R,Pose_F1,Pose_mAP50,Pose_mAP75,Pose_mAP50-95\n")
        
        for r in all_results:
            f.write(f"{r['model_name']},{r['data_yaml']},{r['gflops']:.1f},{r['parameters']/1e6:.2f},")
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
    print("Cross-dataset generalization validation")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Show the experiment list
    print(f"\n{len(MODEL_DATA_MAP)} generalization experiments:")
    for i, (model_name, data_yaml) in enumerate(MODEL_DATA_MAP.items(), 1):
        print(f"  {i}. {model_name}")
        print(f"     dataset: {data_yaml}")

    # Output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate one by one
    all_results = []
    for i, (model_name, data_yaml) in enumerate(MODEL_DATA_MAP.items(), 1):
        print(f"\n[{i}/{len(MODEL_DATA_MAP)}] validating...")
        result = validate_single_model(model_name, BASE_DIR, data_yaml, OUTPUT_DIR)
        if result:
            all_results.append(result)

    # Comparison report
    generate_comparison_report(all_results, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Generalization validation complete.")
    print(f"Validated successfully: {len(all_results)}/{len(MODEL_DATA_MAP)} experiments")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
