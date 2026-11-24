"""Re-run run 1 from scratch and check it matches the saved results exactly.

    python scripts/check_reproducible.py          # run 1, about 40 s
    python scripts/check_reproducible.py --run 2
"""

import argparse
import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
seeds = {1: 10_000_000, 2: 20_000_000, 3: 30_000_000}

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=1, choices=[1, 2, 3])
args = parser.parse_args()
saved = root / f"results/final/run_{args.run}"

with tempfile.TemporaryDirectory() as tmp:
    print(f"Re-running run {args.run} (seed base {seeds[args.run]:,}) into a temporary folder...", flush=True)
    subprocess.run([sys.executable, str(root / "scripts/run_experiment.py"), "--out", tmp,
                    "--train", "2400", "--calibration", "4000", "--test", "2000",
                    "--seed-base", str(seeds[args.run])],
                   cwd=root, check=True, stdout=subprocess.DEVNULL)
    old = json.loads((saved / "metrics.json").read_text())["metrics"]
    new = json.loads((Path(tmp) / "metrics.json").read_text())["metrics"]
    same_predictions = filecmp.cmp(saved / "predictions.csv", Path(tmp) / "predictions.csv", shallow=False)

print(f"\n{'test set':<26}{'method':<28}{'saved recall':>14}{'new recall':>12}")
for test_set, methods in old.items():
    for method, m in methods.items():
        print(f"{test_set:<26}{method:<28}{m['recall']:>14.4f}{new[test_set][method]['recall']:>12.4f}")
print()
print("metrics.json identical:   ", old == new)
print("predictions.csv identical:", same_predictions)
sys.exit(0 if old == new and same_predictions else 1)
