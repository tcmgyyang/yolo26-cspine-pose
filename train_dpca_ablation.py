"""
DPCA-placement ablation — batch training script.

Trains 15 DPCA-placement configurations (positions A/B/C/D and their
combinations, excluding the baseline) on two GPUs in sequence.
"""

from ultralytics import YOLO
import torch
import os

# ==================== Sigma toggle ====================
USE_CUSTOM_SIGMA = True   # True: inject anatomically tuned MS-OKS sigmas
                          # False: keep the uniform-sigma default

# Sigma values (only effective when USE_CUSTOM_SIGMA is True)
SIGMA_CORNER  = 0.05      # vertebra corners (AS/AI/PS/PI) — high precision
SIGMA_SPINOUS = 0.07      # spinous-process tip (SP) — moderate precision
SIGMA_LAMINA  = 0.10      # lamina point (LP) — looser tolerance
# ======================================================

# 15 DPCA placement configurations (baseline excluded)
DPCA_CONFIGS = [
    # Single position (4)
    ('yolo26s-pose-dpca-A.yaml', 'DPCA_A'),
    ('yolo26s-pose-dpca-B.yaml', 'DPCA_B'),
    ('yolo26s-pose-dpca-C.yaml', 'DPCA_C'),
    ('yolo26s-pose-dpca-D.yaml', 'DPCA_D'),
    # Pairwise (6)
    ('yolo26s-pose-dpca-AB.yaml', 'DPCA_AB'),
    ('yolo26s-pose-dpca-AC.yaml', 'DPCA_AC'),
    ('yolo26s-pose-dpca-AD.yaml', 'DPCA_AD'),
    ('yolo26s-pose-dpca-BC.yaml', 'DPCA_BC'),
    ('yolo26s-pose-dpca-BD.yaml', 'DPCA_BD'),
    ('yolo26s-pose-dpca-CD.yaml', 'DPCA_CD'),
    # Triples (4)
    ('yolo26s-pose-dpca-ABC.yaml', 'DPCA_ABC'),
    ('yolo26s-pose-dpca-ABD.yaml', 'DPCA_ABD'),
    ('yolo26s-pose-dpca-ACD.yaml', 'DPCA_ACD'),
    ('yolo26s-pose-dpca-BCD.yaml', 'DPCA_BCD'),
    # All four (1)
    ('yolo26s-pose-dpca-ABCD.yaml', 'DPCA_ABCD'),
]


def get_cervical_sigmas(device='cpu'):
    """Build the 35-element sigma vector for the cervical-spine schema."""
    if not USE_CUSTOM_SIGMA:
        return torch.ones(35, device=device) / 35

    sigmas = []
    # C2: AS, AI, PS, PI, SP (5 points, no LP)
    sigmas.extend([SIGMA_CORNER] * 4)
    sigmas.append(SIGMA_SPINOUS)
    # C3-C7: AS, AI, PS, PI, SP, LP (6 points each)
    for _ in range(5):
        sigmas.extend([SIGMA_CORNER] * 4)
        sigmas.append(SIGMA_SPINOUS)
        sigmas.append(SIGMA_LAMINA)

    return torch.tensor(sigmas, dtype=torch.float32, device=device)


def on_train_start(trainer):
    """Trainer callback that swaps in the custom sigma vector at train start."""
    if not USE_CUSTOM_SIGMA:
        return

    if hasattr(trainer, 'loss') and hasattr(trainer.loss, 'keypoint_loss'):
        device = trainer.loss.keypoint_loss.sigmas.device
        trainer.loss.keypoint_loss.sigmas = get_cervical_sigmas(device)
        print(f"\n[MS-OKS] sigmas injected  "
              f"corner={SIGMA_CORNER}  spinous={SIGMA_SPINOUS}  lamina={SIGMA_LAMINA}\n")


def train_single_config(yaml_file, exp_name):
    """Train a single placement configuration."""
    model_path = f'ultralytics/cfg/models/26/{yaml_file}'

    if not os.path.exists(model_path):
        print(f"[ERROR] config not found: {model_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Starting:   {exp_name}")
    print(f"Config YAML: {yaml_file}")
    print(f"{'='*60}\n")

    model = YOLO(model_path)
    model.add_callback('on_train_start', on_train_start)

    model.train(
        data='dataset/data.yaml',   # edit to your dataset YAML
        imgsz=1280,
        epochs=600,
        batch=16,
        device='1,2',               # dual-GPU
        workers=18,

        # Optimizer & LR schedule
        optimizer='AdamW',
        lr0=0.0005,
        lrf=0.005,
        weight_decay=0.0005,

        # Loss weights
        pose=30.0,
        kobj=5.0,
        box=2.0,

        # Conservative augmentations for clinical X-rays
        degrees=5.0,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.05,

        # Final clean-up phase
        close_mosaic=150,

        # Misc
        val=True,
        amp=False,
        label_smoothing=0.0,
        patience=0,
        project='spine_extreme',
        name=f'{exp_name}_sigma' if USE_CUSTOM_SIGMA else exp_name,
    )

    print(f"\n[DONE] {exp_name} training complete.\n")
    return True


if __name__ == '__main__':
    print("="*60)
    print("DPCA placement ablation — batch training")
    print(f"Configurations to run: {len(DPCA_CONFIGS)}")
    print(f"Device: GPUs 1,2 (dual-GPU)")
    print(f"Custom MS-OKS sigmas: {'enabled' if USE_CUSTOM_SIGMA else 'disabled'}")
    print("="*60)

    results = []

    for i, (yaml_file, exp_name) in enumerate(DPCA_CONFIGS):
        print(f"\n[progress] {i+1}/{len(DPCA_CONFIGS)}")

        try:
            success = train_single_config(yaml_file, exp_name)
            results.append((exp_name, 'SUCCESS' if success else 'FAILED'))
        except Exception as e:
            print(f"[ERROR] {exp_name} training failed: {e}")
            results.append((exp_name, f'ERROR: {e}'))

    # Print summary
    print("\n" + "="*60)
    print("Training summary")
    print("="*60)
    for exp_name, status in results:
        print(f"  {exp_name}: {status}")
    print("="*60)
