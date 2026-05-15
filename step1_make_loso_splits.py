"""
Step 1: From the per-source image lists produced by step0, build the YAML
data configs for both Leave-One-Source-Out (LOSO) and single-source training.

Produces two groups of YAMLs:

    loso_configs/loso/
        loso_test_WJ.yaml   # train = 5 sources, test = WJ
        loso_test_VD.yaml   # ...
        ... 6 files in total
    loso_configs/single_source/
        single_WJ.yaml      # train = WJ, test = WJ (single-source training;
                            # cross-domain val is performed separately by step3)
        ... 6 files in total

YOLO accepts the `train`/`val`/`test` fields as .txt files (one absolute image
path per line). The label path is derived by replacing "/images/" with
"/labels/" in the image path.

Usage:
    python step1_make_loso_splits.py \
        --center_lists ./center_lists \
        --out_dir      ./loso_configs \
        --seed         42
"""
from __future__ import annotations
import argparse, random, yaml
from pathlib import Path

SOURCES = ["WJ", "VD", "GD", "HB", "SX", "TJ"]
NC  = 1
KPT = 35
VAL_FRAC = 0.10   # 10% of the train pool is reserved for early-stopping val


def read_center(list_dir: Path, code: str) -> list[str]:
    p = list_dir / f"{code}_images.txt"
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def write_image_list(paths: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(paths) + "\n", encoding="utf-8")


def write_yaml(train_txt: str, val_txt: str, test_txt: str, out_path: Path) -> None:
    cfg = {
        "train":     train_txt,
        "val":       val_txt,
        "test":      test_txt,
        "nc":        NC,
        "kpt_shape": [KPT, 3],
        "flip_idx":  [],
        "names":     {0: "cervical_spine"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center_lists", required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--seed",         default=42, type=int)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    list_dir = Path(args.center_lists)
    out_dir  = Path(args.out_dir)
    img_list_dir = out_dir / "image_lists"

    by_center: dict[str, list[str]] = {c: read_center(list_dir, c) for c in SOURCES}
    for c in SOURCES:
        print(f"  {c}: {len(by_center[c]):5d} images loaded")

    # --- LOSO: 6 folds -----------------------------------------------------
    print("\n=== LOSO configs ===")
    loso_dir = out_dir / "loso"
    for hold in SOURCES:
        train_pool = []
        for c in SOURCES:
            if c != hold:
                train_pool.extend(by_center[c])
        rng.shuffle(train_pool)
        n_val  = max(50, int(len(train_pool) * VAL_FRAC))
        val_s  = train_pool[:n_val]
        train_s = train_pool[n_val:]
        test_s  = by_center[hold]

        t_txt = img_list_dir / f"loso_{hold}_train.txt"
        v_txt = img_list_dir / f"loso_{hold}_val.txt"
        x_txt = img_list_dir / f"loso_{hold}_test.txt"
        write_image_list(train_s, t_txt)
        write_image_list(val_s,   v_txt)
        write_image_list(test_s,  x_txt)

        print(f"  LOSO test={hold}: train {len(train_s)} / val {len(val_s)} / test {len(test_s)}")
        write_yaml(str(t_txt.resolve()), str(v_txt.resolve()), str(x_txt.resolve()),
                   loso_dir / f"loso_test_{hold}.yaml")

    # --- Single-source training (used to build the 6x6 transfer matrix) ----
    print("\n=== Single-source configs (for transfer matrix) ===")
    single_dir = out_dir / "single_source"
    for c in SOURCES:
        pool = by_center[c][:]
        rng.shuffle(pool)
        n_val  = max(20, int(len(pool) * VAL_FRAC))
        val_s  = pool[:n_val]
        train_s = pool[n_val:]
        # The `test` field points at the source's own val split as a
        # placeholder; true cross-domain evaluation is performed separately
        # by step3 over the other five sources.
        t_txt = img_list_dir / f"single_{c}_train.txt"
        v_txt = img_list_dir / f"single_{c}_val.txt"
        x_txt = img_list_dir / f"single_{c}_test.txt"
        write_image_list(train_s, t_txt)
        write_image_list(val_s,   v_txt)
        write_image_list(val_s,   x_txt)  # placeholder

        print(f"  single {c}: train {len(train_s)} / val {len(val_s)}")
        write_yaml(str(t_txt.resolve()), str(v_txt.resolve()), str(x_txt.resolve()),
                   single_dir / f"single_{c}.yaml")

    print(f"\nDone. YAMLs in {out_dir}")
    print("Next: bash step2_run_loso_train.sh")


if __name__ == "__main__":
    main()
