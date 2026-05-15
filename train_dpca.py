"""
Standalone training script for the DPCA-augmented YOLO26-pose model with
anatomically tuned MS-OKS sigmas. Equivalent to `train_cervical_pose.py`
but pinned to the DPCA-only architecture (no DySample), used as an ablation
control.

Edit the bottom `model.train(...)` call to point at your dataset YAML.
"""

from ultralytics import YOLO
import torch

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


if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/26/yolo26s-pose-dpca.yaml")
    model.add_callback("on_train_start", on_train_start)

    model.train(
        data="dataset/data.yaml",   # edit to your dataset YAML
        imgsz=1280,
        epochs=600,
        batch=16,
        device="0",
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
        project="spine_extreme",
        name="yolo26s_DPCA_sigma" if USE_CUSTOM_SIGMA else "yolo26s_DPCA",
    )
