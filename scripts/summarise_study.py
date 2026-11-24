"""Pool the three runs and the CHIME check into results/final/summary.json."""

import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from frb.experiment import wilson
from frb.detection import Search
from frb.simulation import Config, simulate

folder = root / "results/final"
runs = [json.loads((folder / f"run_{run}/metrics.json").read_text()) for run in range(1, 4)]
methods = ["classical", "classical_dm_contrast", "physics_features_logistic"]
conditions = ["test_familiar", "test_unseen_interference", "test_scattered"]
summary = {"runs": 3, "examples_per_run": 12400, "conditions": {}}

for condition in conditions:
    summary["conditions"][condition] = {}
    for method in methods:
        values = [run["metrics"][condition][method] for run in runs]
        counts = {name: sum(value[name] for value in values) for name in ["tp", "fp", "tn", "fn"]}
        positives = counts["tp"] + counts["fn"]
        negatives = counts["fp"] + counts["tn"]
        summary["conditions"][condition][method] = {
            **counts, "recall": counts["tp"] / positives,
            "false_positive_rate": counts["fp"] / negatives,
            "recall_95ci": wilson(counts["tp"], positives),
            "fpr_95ci": wilson(counts["fp"], negatives),
            "run_recall": [value["recall"] for value in values],
            "run_fpr": [value["false_positive_rate"] for value in values],
            "mean_average_precision": float(np.mean([value["average_precision"] for value in values])),
        }

# all three runs must have used the same source code
for filename in ["simulation.py", "detection.py", "experiment.py"]:
    assert len({run["code_sha256"][filename] for run in runs}) == 1

thresholds = {
    run_number: {method: runs[run_number - 1]["metrics"]["test_familiar"][method]["threshold"] for method in methods}
    for run_number in range(1, 4)
}
# recall in SNR bands of width 2, using each run's own thresholds
snr_rows = []
for condition in conditions:
    for low in [4, 6, 8, 10, 12, 14]:
        for method in methods:
            count = accepted = 0
            for run_number in range(1, 4):
                with (folder / f"run_{run_number}/predictions.csv").open() as handle:
                    for row in csv.DictReader(handle):
                        if row["split"] != condition or row["label"] != "1":
                            continue
                        snr = float(row["injected_template_snr"])
                        if low <= snr < low + 2:
                            count += 1
                            accepted += float(row[method]) > thresholds[run_number][method]
            snr_rows.append({"condition": condition, "method": method, "snr_low": low,
                             "snr_high": low + 2, "count": count, "accepted": accepted,
                             "recall": accepted / count})

with (folder / "recovery_by_snr.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(snr_rows[0]))
    writer.writeheader()
    writer.writerows(snr_rows)

# rough timing of search + classifier on one window
model = joblib.load(folder / "run_1/logistic_model.joblib")
search = Search(Config())
durations = []
examples = [simulate(Config(), 90_000_000 + index, bool(index % 2))[0] for index in range(40)]
search.transform(examples[0])  # warm-up
for observation in examples:
    start = time.perf_counter()
    features, candidate = search.transform(observation)
    model.predict_proba(features[None, :])
    durations.append((time.perf_counter() - start) * 1000)
summary["timing"] = {
    "examples": len(examples), "median_ms": float(np.median(durations)),
    "p95_ms": float(np.percentile(durations, 95)),
}

validation = json.loads((folder / "chime/validation.json").read_text())
summary["chime"] = {key: value for key, value in validation.items() if key != "runs"}
summary["chime"]["semi_synthetic"] = {}
for method in methods:
    values = [run[method] for run in validation["runs"].values()]
    positives = sum(value["tp"] + value["fn"] for value in values)
    negatives = sum(value["fp"] + value["tn"] for value in values)
    tp = sum(value["tp"] for value in values)
    fp = sum(value["fp"] for value in values)
    summary["chime"]["semi_synthetic"][method] = {
        "tp": tp, "fp": fp, "positives": positives, "negatives": negatives,
        "recall": tp / positives, "false_positive_rate": fp / negatives,
    }
(folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

