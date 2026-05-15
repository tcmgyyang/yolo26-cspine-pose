"""
Step 3: Single-source training -> cross-domain evaluation, producing the 6x6
transfer matrix.

Only the proposed model is trained here. For each training source, after
training completes we evaluate the checkpoint on the test split of all six
sources.

Usage:
    python step3_run_transfer_train.py --device 3
    python step3_run_transfer_train.py --device 0 --train_source WJ   # one source
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

PROPOSED_ARCH = "ultralytics/cfg/models/26/yolo26s-pose-dpca-AC.yaml"
YAML_SINGLE   = Path("./loso_configs/single_source")
YAML_LOSO     = Path("./loso_configs/loso")
SOURCES       = ["WJ", "VD", "GD", "HB", "SX", "TJ"]

SIGMA_CORNER  = 0.05
SIGMA_SPINOUS = 0.07
SIGMA_LAMINA  = 0.10


def get_cervical_sigmas(device: str = "cpu") -> torch.Tensor:
    sigmas: list[float] = []
    sigmas.extend([SIGMA_CORNER] * 4); sigmas.append(SIGMA_SPINOUS)   # C2
    for _ in range(5):                                                 # C3-C7
        sigmas.extend([SIGMA_CORNER] * 4); sigmas.append(SIGMA_SPINOUS); sigmas.append(SIGMA_LAMINA)
    return torch.tensor(sigmas, dtype=torch.float32, device=device)


def make_sigma_callback():
    def on_train_start(trainer):
        if hasattr(trainer, "loss") and hasattr(trainer.loss, "keypoint_loss"):
            dev = trainer.loss.keypoint_loss.sigmas.device
            trainer.loss.keypoint_loss.sigmas = get_cervical_sigmas(dev)
            print(f"[MS-OKS] sigma injected")
    return on_train_start


TRAIN_KW = dict(
    imgsz=1280, epochs=600, batch=16, workers=4,
    optimizer="AdamW",
    lr0=0.0005, lrf=0.005, weight_decay=0.0005,
    pose=30.0, kobj=5.0, box=2.0,
    degrees=5.0, scale=0.3, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0, mosaic=1.0, mixup=0.05,
    close_mosaic=150,
    val=True, amp=False,
    label_smoothing=0.0, patience=0,
    seed=42,
)


def train_one_center(train_s: str, device: str) -> None:
    data_yaml = YAML_SINGLE / f"single_{train_s}.yaml"
    project   = "runs/transfer"
    name      = f"train_{train_s}"

    print(f"\n{'=' * 60}")
    print(f"  Transfer: single-source training on {train_s}")
    print(f"{'=' * 60}")

    model = YOLO(PROPOSED_ARCH)
    model.add_callback("on_train_start", make_sigma_callback())
    model.train(
        data=str(data_yaml),
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        **TRAIN_KW,
    )

    # Evaluate on every source's test split.
    # The vendored Ultralytics auto-prepends "runs/pose/" to user-supplied
    # `project` paths, so we look up `best.pt` under that prefix.
    best = f"runs/pose/{project}/{name}/weights/best.pt"
    for test_s in SOURCES:
        print(f"\n  val: train={train_s}  ->  test={test_s}")
        eval_yaml = YAML_LOSO / f"loso_test_{test_s}.yaml"
        YOLO(best).val(
            data=str(eval_yaml),
            device=device,
            split="test",
            project=project,
            name=f"train_{train_s}_test_{test_s}",
            exist_ok=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",       default="0")
    ap.add_argument("--train_source", nargs="+", default=SOURCES,
                    help="subset of training sources to run")
    args = ap.parse_args()

    for c in args.train_source:
        train_one_center(c, args.device)

    print("\nTransfer done. Next: python step4_collect_results.py --runs_dir ./runs")


if __name__ == "__main__":
    main()
