"""
Step 4: Walk the YOLO `runs/` tree, collect every mAP value, and print
Python literals ready to paste into `fig4_generalization.py`.

Usage:
    python step4_collect_results.py                      # default --runs_dir ./runs/pose/runs
    python step4_collect_results.py --runs_dir ./runs    # legacy path

Read order per run directory:
    1) results.json
    2) results.csv
    3) fallback: parse the most recent "all ..." line for that tag in
       runs/_rerun_logs/*.log

Output mAP unit: percent (multiplied by 100).
"""
import argparse, csv, json, re
from pathlib import Path
import numpy as np

SOURCES = ["WJ", "TJ", "HB", "SX", "GD", "VD"]
LOG_DIR = Path("runs/_rerun_logs")


def _parse_log_for_tag(tag: str) -> tuple[float, float]:
    """Locate the most recent (Pose mAP50-95, Pose mAP50) values for the given
    tag in `_rerun_logs/*.log`. Returns values in percent.

    Two log formats are supported:
      A. _rerun_loso_val.py: "EVAL  baseline/test_WJ  on held-out test split"
         example tag: "loso_baseline/test_WJ_eval"
      B. step3: "val: train=WJ  -> test=GD"
         example tag: "transfer/train_WJ_test_GD"
    """
    if not LOG_DIR.exists():
        return float("nan"), float("nan")

    patterns = []

    # Pattern A: EVAL <name> on held-out test split
    short_a   = tag.replace("loso_", "").replace("_eval", "")           # baseline/test_WJ
    short_alt = tag.replace("transfer/", "")                            # train_WJ_test_GD
    patterns.append(re.compile(
        r"EVAL\s+(?:" + re.escape(short_a) + r"|" + re.escape(short_alt) +
        r")\b.*?^\s*all\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)",
        re.DOTALL | re.MULTILINE,
    ))

    # Pattern B: step3 transfer val output "val: train=X  -> test=Y"
    m = re.match(r"transfer/train_(\w+)_test_(\w+)", tag)
    if m:
        train_s, test_s = m.group(1), m.group(2)
        # Step3 may write the arrow as either "->" (ASCII) or U+2192.
        patterns.append(re.compile(
            r"val:\s+train=" + re.escape(train_s) + r"\s+(?:->|->)\s+test=" + re.escape(test_s) +
            r".*?^\s*all\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)",
            re.DOTALL | re.MULTILINE,
        ))

    last50, last5095 = float("nan"), float("nan")
    for log in sorted(LOG_DIR.glob("*.log")):
        text = log.read_text(errors="ignore")
        for pat in patterns:
            for m in pat.finditer(text):
                try:
                    last50   = float(m.group(1)) * 100
                    last5095 = float(m.group(2)) * 100
                except ValueError:
                    pass
    return last5095, last50


def read_map_from_val(run_dir: Path) -> tuple[float, float]:
    """Return (mAP@50:95, mAP@50) in percent. Precedence: results.json > results.csv > log parse."""
    json_path = run_dir / "results.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            map5095 = map50 = float("nan")
            for k, v in data.items():
                if "50-95" in k or "50:95" in k:
                    map5095 = float(v)
                elif "50" in k and "95" not in k and "50-95" not in k:
                    map50 = float(v)
            return map5095, map50
        except Exception:
            pass

    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open()))
        if rows:
            last = rows[-1]
            cands_5095 = ["metrics/mAP50-95(P)", "metrics/mAP50-95(B)",
                          "pose_mAP50-95", "val/pose_mAP50-95"]
            cands_50   = ["metrics/mAP50(P)",    "metrics/mAP50(B)",
                          "pose_mAP50",       "val/pose_mAP50"]
            map5095 = next((float(last[c]) for c in cands_5095
                            if c in last and last[c].strip()), float("nan"))
            map50   = next((float(last[c]) for c in cands_50
                            if c in last and last[c].strip()), float("nan"))
            return map5095 * 100, map50 * 100

    # Fallback: parse the rerun logs.
    # Example run_dir: runs/pose/runs/loso_baseline/test_WJ_eval
    # Extracted tag:   "loso_baseline/test_WJ_eval" or "transfer/train_WJ_test_GD"
    parts = run_dir.parts
    # Find the leading "loso_*/..." or "transfer/..." sub-path
    for i, p in enumerate(parts):
        if p.startswith("loso_") or p == "transfer":
            tag = "/".join(parts[i:])
            v5095, v50 = _parse_log_for_tag(tag)
            if not (np.isnan(v5095) and np.isnan(v50)):
                return v5095, v50
            break

    print(f"  [WARN] no results found in {run_dir}")
    return float("nan"), float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="./runs/pose/runs",
                    help="vendored Ultralytics writes to runs/pose/runs/; legacy step4 default is ./runs")
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    # --- LOSO -------------------------------------------------------------
    loso_base_5095, loso_base_50   = [], []
    loso_prop_5095, loso_prop_50   = [], []

    for c in SOURCES:
        b_dir = runs / "loso_baseline" / f"test_{c}_eval"
        p_dir = runs / "loso_proposed" / f"test_{c}_eval"
        b5095, b50 = read_map_from_val(b_dir)
        p5095, p50 = read_map_from_val(p_dir)
        loso_base_5095.append(round(b5095, 2))
        loso_base_50  .append(round(b50,   2))
        loso_prop_5095.append(round(p5095, 2))
        loso_prop_50  .append(round(p50,   2))

    # --- Transfer matrix ---------------------------------------------------
    transfer = []
    for train_s in SOURCES:
        row = []
        for test_s in SOURCES:
            t_dir = runs / "transfer" / f"train_{train_s}_test_{test_s}"
            val, _ = read_map_from_val(t_dir)
            row.append(round(val, 2))
        transfer.append(row)

    # --- Output ------------------------------------------------------------
    print("\n# --- Paste into fig4_generalization.py (replaces PLACEHOLDER) ---")
    print(f"LOSO_BASE    = np.array({loso_base_5095})   # mAP@50:95 baseline")
    print(f"LOSO_PROP    = np.array({loso_prop_5095})   # mAP@50:95 proposed")
    print(f"LOSO_BASE_50 = np.array({loso_base_50})   # mAP@50 baseline")
    print(f"LOSO_PROP_50 = np.array({loso_prop_50})   # mAP@50 proposed")
    print()
    print("TRANSFER = np.array([")
    for i, row in enumerate(transfer):
        print(f"    {row},   # train={SOURCES[i]}")
    print("])")


if __name__ == "__main__":
    main()
