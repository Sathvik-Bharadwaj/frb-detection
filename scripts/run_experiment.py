"""Run a single experiment. Example: python scripts/run_experiment.py --out results/smoke"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frb.experiment import main

if __name__ == "__main__":
    main()
