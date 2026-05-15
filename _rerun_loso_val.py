"""
Re-run the held-out test-split evaluation for all 12 LOSO `best.pt` checkpoints.
Equivalent to the `model.val(split="test")` block at the end of
`step2_run_loso_train.py`, but evaluation-only — no retraining.

NOTE: This is a workaround. The inline val step in step2 can fail when the
vendored Ultralytics auto-prepends "runs/pose/" to the `project` path it was
given, which causes `best.pt` to be searched in the un-prefixed location.
The proper fix is to make step2's val step path-aware; until that lands,
this script recovers the missing evaluation outputs.
TODO: remove once the root cause is fixed upstream of step2.

Usage: python _rerun_loso_val.py --device 0
"""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO

YAML_DIR = Path("./loso_configs/loso")
SOURCES  = ["WJ", "VD", "GD", "HB", "SX", "TJ"]
TAGS     = ["baseline", "proposed"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    for tag in TAGS:
        for c in SOURCES:
            project = f"runs/loso_{tag}"
            name    = f"test_{c}"
            best    = Path(f"{project}/{name}/weights/best.pt")
            data    = YAML_DIR / f"loso_test_{c}.yaml"

            full_best = Path(f"runs/pose/{project}/{name}/weights/best.pt")
            if not full_best.exists():
                print(f"[skip] best.pt not found: {full_best}")
                continue

            print(f"\n{'='*60}\n  EVAL  {tag}/test_{c}  on held-out test split\n{'='*60}")
            YOLO(str(full_best)).val(
                data=str(data),
                device=args.device,
                split="test",
                project=project,
                name=f"{name}_eval",
                exist_ok=True,
            )

    print("\nALL 12 LOSO test-split evaluations done.")


if __name__ == "__main__":
    main()
