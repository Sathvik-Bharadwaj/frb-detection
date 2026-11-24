"""Why does the learned score miss CHIME-shaped pulses?

Compares the search features of simulated test pulses with those of the
CHIME-shaped injections (same seeds as validate_chime.py) and saves the
medians next to the model coefficients.
"""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from frb.detection import FEATURE_NAMES, Search
from frb.observations import fitted_profile, read_chime
from frb.simulation import Config, dispersion_delay, simulate

cfg = Config()
search = Search(cfg)
observation = read_chime(root / "data/raw/FRB20180725A_waterfall.h5")
relative_time, profiles, available = fitted_profile(observation, cfg)

# shape of the fitted CHIME pulse after binning to 32 channels
total = profiles.sum(axis=0)
step_ms = float(np.median(np.diff(observation.times_s)) * 1000)
energy = profiles.sum(axis=1) / profiles.sum()
shape = {"fwhm_ms": float((total >= total.max() / 2).sum() * step_ms),
         "energy_fraction_600_700MHz": float(energy[: cfg.nchan // 2].sum()),
         "channels_with_data": int(available.sum())}

chime, familiar, negatives = [], [], []
for run in range(1, 4):
    seed_start = 60_000_000 + run * 1_000_000
    for index in range(1, 800, 2):  # odd indices are the positives
        seed = seed_start + index
        data, _ = simulate(cfg, seed, False)
        rng = np.random.default_rng(seed + 100_000_000)
        dm, arrival, snr = rng.uniform(24, 160), rng.uniform(0.24, 0.64), rng.uniform(4, 16)
        delay = dispersion_delay(dm, cfg.frequencies)
        template = np.zeros_like(data)
        for channel in range(cfg.nchan):
            template[channel] = np.interp(cfg.times - arrival - delay[channel], relative_time,
                                          profiles[channel], left=0, right=0)
        data += snr * template / np.linalg.norm(template)
        chime.append(search.transform(data)[0])
    saved = np.load(root / f"results/final/run_{run}/test_familiar_features.npz")
    familiar.append(saved["features"][saved["labels"] == 1])
    negatives.append(saved["features"][saved["labels"] == 0])

chime = np.array(chime)
familiar = np.concatenate(familiar)
negatives = np.concatenate(negatives)
coefficients = [json.loads((root / f"results/final/run_{run}/metrics.json").read_text())
                ["standardised_coefficients"] for run in range(1, 4)]

table = {}
for j, name in enumerate(FEATURE_NAMES):
    table[name] = {"mean_coefficient": float(np.mean([c[name] for c in coefficients])),
                   "median_simulated_pulse": float(np.median(familiar[:, j])),
                   "median_chime_pulse": float(np.median(chime[:, j])),
                   "median_no_pulse": float(np.median(negatives[:, j]))}
widths = {f"{int(w)}": {"simulated": float(np.mean(familiar[:, 5] == w)),
                        "chime": float(np.mean(chime[:, 5] == w))} for w in search.widths}

out = {"chime_profile": shape, "features": table, "winning_width_fraction": widths}
(root / "results/final/chime/feature_comparison.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
