"""
Step 2: LOSO training — 6 baseline runs + 6 proposed runs.

Methodological constraint:
  Both models are initialised from scratch from the architecture YAMLs; we do
  NOT warm-start from a `best.pt` previously trained on WJ + VD. Doing so
  would leak the LOSO-WJ / LOSO-VD test domain into pre-training.

Training hyper-parameters match those of the original
`spine_extreme/ablation/DPCA_AC_sigma` configuration.

Usage:
    # Run all 12 (default)
    python step2_run_loso_train.py --device 3

    # Run only the proposed model (if baselines already exist)
    python step2_run_loso_train.py --device 3 --skip_baseline

    # Run only specific source folds
    python step2_run_loso_train.py --device 3 --sources WJ VD
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# --- Architecture YAMLs (adjust BASELINE_ARCH if a different baseline is wanted)
BASELINE_ARCH = "ultralytics/cfg/models/26/yolo26s-pose.yaml"
PROPOSED_ARCH = "ultralytics/cfg/models/26/yolo26s-pose-dpca-AC.yaml"

YAML_DIR = Path("./loso_configs/loso")
SOURCES  = ["WJ", "VD", "GD", "HB", "SX", "TJ"]

# --- MS-OKS custom sigmas (used by the proposed model) --------------------
SIGMA_CORNER  = 0.05   # vertebra corners (AS/AI/PS/PI)
SIGMA_SPINOUS = 0.07   # spinous-process tip (SP)
SIGMA_LAMINA  = 0.10   # lamina point (LP)


def get_cervical_sigmas(device: str = "cpu") -> torch.Tensor:
    sigmas: list[float] = []
    # C2: 4 corners + 1 spinous (no LP)
    sigmas.extend([SIGMA_CORNER] * 4)
    sigmas.append(SIGMA_SPINOUS)
    # C3–C7: 4 corners + 1 spinous + 1 lamina
    for _ in range(5):
        sigmas.extend([SIGMA_CORNER] * 4)
        sigmas.append(SIGMA_SPINOUS)
        sigmas.append(SIGMA_LAMINA)
    return torch.tensor(sigmas, dtype=torch.float32, device=device)


def make_sigma_callback():
    def on_train_start(trainer):
        if hasattr(trainer, "loss") and hasattr(trainer.loss, "keypoint_loss"):
            dev = trainer.loss.keypoint_loss.sigmas.device
            trainer.loss.keypoint_loss.sigmas = get_cervical_sigmas(dev)
            print(f"\n[MS-OKS] sigma injected  "
                  f"corner={SIGMA_CORNER}  spinous={SIGMA_SPINOUS}  lamina={SIGMA_LAMINA}\n")
    return on_train_start


# --- Shared training hyper-parameters (match the DPCA_AC_sigma reference run)
TRAIN_KW = dict(
    imgsz=1280,
    epochs=600,
    batch=16,
    workers=18,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.005,
    weight_decay=0.0005,
    pose=30.0,
    kobj=5.0,
    box=2.0,
    degrees=5.0,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=1.0,
    mixup=0.05,
    close_mosaic=150,
    val=True,
    amp=False,
    label_smoothing=0.0,
    patience=0,
    seed=42,
)


def run_one(tag: str, arch_yaml: str, inject_sigma: bool,
            hold_source: str, device: str) -> None:
    data_yaml = YAML_DIR / f"loso_test_{hold_source}.yaml"
    project   = f"runs/loso_{tag}"
    name      = f"test_{hold_source}"

    print(f"\n{'=' * 60}")
    print(f"  {tag.upper()}  |  LOSO test = {hold_source}")
    print(f"{'=' * 60}")

    model = YOLO(arch_yaml)
    if inject_sigma:
        model.add_callback("on_train_start", make_sigma_callback())

    model.train(
        data=str(data_yaml),
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        **TRAIN_KW,
    )

    # Evaluate on the held-out source's test split
    best = f"{project}/{name}/weights/best.pt"
    YOLO(best).val(
        data=str(data_yaml),
        device=device,
        split="test",
        project=project,
        name=f"{name}_eval",
        exist_ok=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",         default="0")
    ap.add_argument("--sources",        nargs="+", default=SOURCES,
                    help="subset of sources to run")
    ap.add_argument("--skip_baseline",  action="store_true")
    ap.add_argument("--skip_proposed",  action="store_true")
    args = ap.parse_args()

    if not args.skip_baseline:
        print("\n>>> LOSO baseline (stock YOLO26-pose, no sigma injection) <<<")
        for c in args.sources:
            run_one("baseline", BASELINE_ARCH, False, c, args.device)

    if not args.skip_proposed:
        print("\n>>> LOSO proposed (DPCA + DySample + MS-OKS) <<<")
        for c in args.sources:
            run_one("proposed", PROPOSED_ARCH, True, c, args.device)

    print("\nLOSO done. Next: python step3_run_transfer_train.py")


if __name__ == "__main__":
    main()
