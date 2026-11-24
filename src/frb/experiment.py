"""Train the logistic score, calibrate thresholds and evaluate one run."""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .detection import FEATURE_NAMES, Search
from .simulation import Config, simulate


def threshold_for_fpr(negative_scores, target=0.01):
    """Threshold such that at most `target` of the negatives score strictly above it."""
    if not 0 <= target < 1:
        raise ValueError("FPR must lie in [0, 1)")
    v = np.sort(np.asarray(negative_scores))
    if not len(v):
        raise ValueError("Calibration requires negative examples")
    allowed = int(np.floor(target * len(v)))
    return float(v[max(0, len(v) - allowed - 1)])


def wilson(k, n):
    """95% Wilson score interval for k successes out of n."""
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def summarise(labels, scores, threshold):
    y = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) > threshold
    tp = int(np.sum(pred & y))
    fn = int(np.sum(~pred & y))
    fp = int(np.sum(pred & ~y))
    tn = int(np.sum(~pred & ~y))
    return {"threshold": threshold, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn), "false_positive_rate": fp / (fp + tn),
            "recall_95ci": wilson(tp, tp + fn), "fpr_95ci": wilson(fp, fp + tn),
            "precision": tp / max(1, tp + fp),
            "average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores))}


def build_split(cfg, search, name, n, base_seed, distribution):
    features, labels, metadata = [], [], []
    start = time.perf_counter()
    for i in range(n):
        # odd indices get a pulse, so every split is balanced
        x, meta = simulate(cfg, base_seed + i, i % 2 == 1, distribution)
        feature, _ = search.transform(x)
        features.append(feature)
        labels.append(meta["label"])
        meta.update(example_id=f"{name}-{i:05d}", split=name)
        metadata.append(meta)
    print(f"{name}: {n} examples in {time.perf_counter() - start:.1f} s", flush=True)
    return np.asarray(features), np.asarray(labels), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/run")
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--calibration", type=int, default=600)
    parser.add_argument("--test", type=int, default=600)
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args()
    if any(n < 20 or n % 2 for n in [args.train, args.calibration, args.test]):
        parser.error("Each split size must be even and at least 20")
    if max(args.train, args.calibration, args.test) >= 1_000_000:
        parser.error("Split sizes must be below 1,000,000 so seed blocks do not overlap")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    search = Search(cfg)

    # each split gets its own block of one million seeds
    definitions = [("train", args.train, args.seed_base + 1_000_000, "familiar"),
                   ("calibration", args.calibration, args.seed_base + 2_000_000, "familiar"),
                   ("test_familiar", args.test, args.seed_base + 3_000_000, "familiar"),
                   ("test_unseen_interference", args.test, args.seed_base + 4_000_000, "unseen_interference"),
                   ("test_scattered", args.test, args.seed_base + 5_000_000, "scattered")]
    splits = {name: build_split(cfg, search, name, n, seed, dist)
              for name, n, seed, dist in definitions}

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=17))
    model.fit(splits["train"][0], splits["train"][1])
    joblib.dump(model, out / "logistic_model.joblib")
    coefficients = dict(zip(FEATURE_NAMES, model[-1].coef_[0].tolist()))

    # thresholds are set on calibration negatives only (target: 1% false positives)
    cal_x, cal_y, _ = splits["calibration"]
    thresholds = {
        "classical": threshold_for_fpr(cal_x[cal_y == 0, 0]),
        "classical_dm_contrast": threshold_for_fpr(cal_x[cal_y == 0, 3]),
        "physics_features_logistic": threshold_for_fpr(model.predict_proba(cal_x[cal_y == 0])[:, 1]),
    }

    results = {}
    score_rows = []
    for name, (x, y, metadata) in splits.items():
        np.savez_compressed(out / f"{name}_features.npz", features=x, labels=y)
        if not name.startswith("test"):
            continue
        scores = {"classical": x[:, 0], "classical_dm_contrast": x[:, 3],
                  "physics_features_logistic": model.predict_proba(x)[:, 1]}
        results[name] = {method: summarise(y, score, thresholds[method])
                         for method, score in scores.items()}
        for i, meta in enumerate(metadata):
            score_rows.append({**meta, **{k: float(v[i]) for k, v in scores.items()}})

    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    with (out / "manifest.jsonl").open("w") as f:
        for _, _, metadata in splits.values():
            for row in metadata:
                f.write(json.dumps(row) + "\n")

    import scipy
    import sklearn
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob("*.py"))}
    report = {"config": cfg.to_dict(), "feature_names": FEATURE_NAMES,
              "target_calibration_fpr": 0.01,
              "window_length_s": cfg.ntime * cfg.dt_s,
              "splits": [dict(name=n, count=k, seed_base=s, distribution=d)
                         for n, k, s, d in definitions],
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scikit_learn": sklearn.__version__, "scipy": scipy.__version__},
              "code_sha256": code_hashes, "standardised_coefficients": coefficients,
              "metrics": results}
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
