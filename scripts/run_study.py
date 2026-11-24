"""Run the three final experiment runs (about a minute each)."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / "results" / "final"
seeds = [10_000_000, 20_000_000, 30_000_000]

for run_number, seed in enumerate(seeds, start=1):
    command = [
        sys.executable, str(root / "scripts" / "run_experiment.py"),
        "--out", str(output / f"run_{run_number}"),
        "--train", "2400", "--calibration", "4000", "--test", "2000",
        "--seed-base", str(seed),
    ]
    subprocess.run(command, cwd=root, check=True)
