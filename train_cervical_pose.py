"""
Cervical-spine pose training (single-file convenience script).

This is a self-contained variant of step2/step3 for ad-hoc training of the
proposed model on a single dataset YAML — useful for sanity checks or
hyper-parameter probes that don't fit the 6-fold LOSO / single-source
pipelines.

Usage:
    python train_cervical_pose.py \\
        --data    loso_configs/single_source/single_WJ.yaml \\
        --device  0
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# ==================== Sigma toggle ====================
USE_CUSTOM_SIGMA = True   # True: inject anatomically tuned MS-OKS sigmas
                          # False: keep the uniform-sigma default

# Sigma values (only effective when USE_CUSTOM_SIGMA is True)
SIGMA_CORNER  = 0.05      # vertebra corners (AS/AI/PS/PI) — high precision
SIGMA_SPINOUS = 0.07      # spinous-process tip (SP) — moderate precision
SIGMA_LAMINA  = 0.10      # lamina point (LP) — looser tolerance
# ======================================================


def get_cervical_sigmas(device: str = "cpu") -> torch.Tensor:
    """Build the 35-element sigma vector for the cervical-spine schema."""
    if not USE_CUSTOM_SIGMA:
        return torch.ones(35, device=device) / 35

    sigmas: list[float] = []
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
    if hasattr(trainer, "loss") and hasattr(trainer.loss, "keypoint_loss"):
        device = trainer.loss.keypoint_loss.sigmas.device
        trainer.loss.keypoint_loss.sigmas = get_cervical_sigmas(device)
        print(
            f"\n[MS-OKS] custom sigmas injected  "
            f"corner={SIGMA_CORNER}  spinous={SIGMA_SPINOUS}  lamina={SIGMA_LAMINA}\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True,
                   help="Dataset YAML, e.g. loso_configs/single_source/single_WJ.yaml")
    p.add_argument("--arch",
                   default="ultralytics/cfg/models/26/yolo26s-pose-dpca-AC.yaml",
                   help="Architecture YAML (default: proposed DPCA-AC).")
    p.add_argument("--device", default="0",
                   help="CUDA device id(s), comma-separated (e.g. '0' or '0,1,2,3').")
    p.add_argument("--project", default="runs/single_run")
    p.add_argument("--name", default=None,
                   help="Run name (default derived from --arch and sigma toggle).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"data YAML not found: {data_path}. "
            "Run step0 + step1 first to generate the LOSO / single-source configs."
        )

    name = args.name or (
        f"{Path(args.arch).stem}_{'sigma' if USE_CUSTOM_SIGMA else 'uniform'}"
    )

    model = YOLO(args.arch)
    model.add_callback("on_train_start", on_train_start)

    model.train(
        data=str(data_path),
        imgsz=1280,
        epochs=600,
        batch=16,
        device=args.device,
        workers=18,

        # Optimizer & LR schedule
        optimizer="AdamW",
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
        seed=42,
        project=args.project,
        name=name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
