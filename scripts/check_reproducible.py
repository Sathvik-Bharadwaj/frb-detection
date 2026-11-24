"""Re-run one of the final runs from scratch and compare it with the saved results.

    python scripts/check_reproducible.py          # run 1, about 40 s
    python scripts/check_reproducible.py --run 2

Three levels are checked:
1. bit for bit: the output files are byte-identical (expected on the same kind of computer);
2. decisions: every accept/reject decision and every TP/FP/TN/FN count is the same;
3. numbers: the largest difference in any score, threshold or metric.

Different processors (for example Apple silicon and an Intel/AMD cloud server) can
round the last digits of floating point sums differently, so level 1 may fail across
machines while levels 2 and 3 still show the result is reproduced.
"""

import argparse
import csv
import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
seeds = {1: 10_000_000, 2: 20_000_000, 3: 30_000_000}
METHODS = ["classical", "classical_dm_contrast", "physics_features_logistic"]
TOLERANCE = 1e-6


def compare_json(a, b, path=""):
    """Return (list of exact mismatches in integers/strings, largest float difference)."""
    mismatches, largest = [], 0.0
    if isinstance(a, dict):
        for key in a:
            m, d = compare_json(a[key], b.get(key), f"{path}/{key}")
            mismatches += m
            largest = max(largest, d)
    elif isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            m, d = compare_json(x, y, f"{path}[{i}]")
            mismatches += m
            largest = max(largest, d)
    elif isinstance(a, float) or isinstance(b, float):
        largest = abs(float(a) - float(b))
    elif a != b:
        mismatches.append(path)
    return mismatches, largest


def read_predictions(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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
    new_dir = Path(tmp)
    old = json.loads((saved / "metrics.json").read_text())["metrics"]
    new = json.loads((new_dir / "metrics.json").read_text())["metrics"]
    bitwise = (old == new) and filecmp.cmp(saved / "predictions.csv", new_dir / "predictions.csv", shallow=False)
    old_rows = read_predictions(saved / "predictions.csv")
    new_rows = read_predictions(new_dir / "predictions.csv")

# level 2: counts and per-example decisions
count_mismatches, metric_diff = compare_json(old, new)
decision_changes = 0
score_diff = 0.0
for o, n in zip(old_rows, new_rows):
    for method in METHODS:
        threshold_old = old[o["split"]][method]["threshold"]
        threshold_new = new[n["split"]][method]["threshold"]
        decision_changes += (float(o[method]) > threshold_old) != (float(n[method]) > threshold_new)
        score_diff = max(score_diff, abs(float(o[method]) - float(n[method])))
same_examples = len(old_rows) == len(new_rows) and all(
    o["example_id"] == n["example_id"] and o["seed"] == n["seed"] for o, n in zip(old_rows, new_rows))

print(f"\n{'test set':<26}{'method':<28}{'saved recall':>14}{'new recall':>12}")
for test_set, methods in old.items():
    for method, m in methods.items():
        print(f"{test_set:<26}{method:<28}{m['recall']:>14.4f}{new[test_set][method]['recall']:>12.4f}")

print()
print(f"Same {len(old_rows):,} test examples (same seeds):  {same_examples}")
print(f"TP/FP/TN/FN counts identical:          {not count_mismatches}")
print(f"Accept/reject decisions that changed:  {decision_changes} of {len(old_rows) * len(METHODS):,}")
print(f"Largest difference in any score:       {score_diff:.1e}")
print(f"Largest difference in any metric:      {metric_diff:.1e}")
print(f"Bit for bit identical files:           {bitwise}")
print()

reproduced = same_examples and not count_mismatches and decision_changes == 0 \
    and max(score_diff, metric_diff) < TOLERANCE
if bitwise:
    print("RESULT: identical, bit for bit.")
elif reproduced:
    print("RESULT: reproduced. Every decision and count is identical; the scores differ only in the "
          "last digits because this processor rounds floating point sums slightly differently.")
else:
    print("RESULT: results differ.")
sys.exit(0 if reproduced or bitwise else 1)
