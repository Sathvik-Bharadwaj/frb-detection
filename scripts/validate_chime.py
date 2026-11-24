"""CHIME checks.

1. Rebuild the time profile of FRB20180725A from the masked waterfall and
   compare its peak with the released time series.
2. Transfer test: take the published fitted profile, re-disperse it at a
   random DM and inject it into fresh simulated backgrounds, then score it
   with each run's saved model and thresholds (no retraining).
"""

import csv
import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from frb.observations import read_chime, fitted_profile
from frb.simulation import Config, dispersion_delay, simulate
from frb.detection import Search
from frb.experiment import summarise

path = root / "data/raw/FRB20180725A_waterfall.h5"
observation = read_chime(path)
settings = Config()
search = Search(settings)
relative_time, profiles, available = fitted_profile(observation, settings)
profile_counts = np.isfinite(observation.intensity).sum(axis=0)
measured_profile = np.divide(
    np.nansum(observation.intensity, axis=0), profile_counts,
    out=np.zeros(len(profile_counts)), where=profile_counts > 0,
)
with h5py.File(path, "r") as handle:
    released_profile = handle["frb/ts"][()]
measured_peak = int(np.argmax(measured_profile))
reference_peak = int(np.argmax(released_profile))
summary = {
    "event": observation.name,
    "catalogue_dm": observation.catalogue_dm,
    "shape": list(observation.intensity.shape),
    "time_step_ms": float(np.median(np.diff(observation.times_s)) * 1000),
    "missing_fraction": float(np.mean(~np.isfinite(observation.intensity))),
    "measured_peak_ms": float(observation.times_s[measured_peak] * 1000),
    "released_peak_ms": float(observation.times_s[reference_peak] * 1000),
    "peak_agreement_samples": abs(measured_peak - reference_peak),
    "model_channels_available": int(available.sum()),
    "runs": {},
}
destination = root / "results/final/chime"
destination.mkdir(parents=True, exist_ok=True)

for run in range(1, 4):
    folder = root / f"results/final/run_{run}"
    model = joblib.load(folder / "logistic_model.joblib")
    experiment = json.loads((folder / "metrics.json").read_text())
    seed_start = 60_000_000 + run * 1_000_000
    rows = []
    for index in range(800):
        positive = bool(index % 2)
        seed = seed_start + index
        # background only, then add the CHIME-shaped pulse for positives
        data, metadata = simulate(settings, seed, False)
        rng = np.random.default_rng(seed + 100_000_000)
        metadata.update(label=int(positive), seed=seed, split=f"real_profile_run_{run}",
                        injected_template_snr=None, dm=None)
        if positive:
            dm = float(rng.uniform(24, 160))
            arrival = float(rng.uniform(0.24, 0.64))
            snr = float(rng.uniform(4, 16))
            delay = dispersion_delay(dm, settings.frequencies)
            template = np.zeros_like(data)
            for channel in range(settings.nchan):
                offset = settings.times - arrival - delay[channel]
                template[channel] = np.interp(offset, relative_time, profiles[channel], left=0, right=0)
            template /= np.linalg.norm(template)
            data += snr * template
            metadata.update(injected_template_snr=snr, dm=dm, arrival_s=arrival)
        features, candidate = search.transform(data)
        rows.append({
            **metadata, "classical": float(features[0]),
            "classical_dm_contrast": float(features[3]),
            "physics_features_logistic": float(model.predict_proba(features[None, :])[0, 1]),
        })
    labels = np.array([row["label"] for row in rows])
    metrics = {}
    for method in ["classical", "classical_dm_contrast", "physics_features_logistic"]:
        threshold = experiment["metrics"]["test_familiar"][method]["threshold"]
        scores = np.array([row[method] for row in rows])
        metrics[method] = summarise(labels, scores, threshold)
    summary["runs"][f"run_{run}"] = metrics
    with (destination / f"predictions_{run}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CHIME transfer test: run {run} done", flush=True)

(destination / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

