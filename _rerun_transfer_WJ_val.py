"""
Re-run the cross-source evaluation for the `train_WJ` transfer checkpoint
across all six test sources (recovers the evaluations missed by step3).

NOTE: Workaround for the same path-prefix issue handled by
`_rerun_loso_val.py` — the inline val step in step3 fails when the vendored
Ultralytics auto-prepends "runs/pose/" to `project`. This script bypasses the
bug by re-running val on the already-trained `best.pt`.
TODO: remove once the root cause is fixed upstream of step3.

Usage: python _rerun_transfer_WJ_val.py --device 0
"""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO

YAML_LOSO = Path("./loso_configs/loso")
SOURCES   = ["WJ", "VD", "GD", "HB", "SX", "TJ"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    train_s = "WJ"
    project = "runs/transfer"
    best    = f"runs/pose/{project}/train_{train_s}/weights/best.pt"

    if not Path(best).exists():
        print(f"[ERROR] best.pt not found: {best}")
        return

    for test_s in SOURCES:
        print(f"\n{'='*60}\n  val: train={train_s}  ->  test={test_s}\n{'='*60}")
        YOLO(best).val(
            data=str(YAML_LOSO / f"loso_test_{test_s}.yaml"),
            device=args.device,
            split="test",
            project=project,
            name=f"train_{train_s}_test_{test_s}",
            exist_ok=True,
        )

    print("\nWJ -> 6 sources val all done.")


if __name__ == "__main__":
    main()
