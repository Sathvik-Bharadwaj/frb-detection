"""Incoherent dedispersion, boxcar search and candidate features."""

import numpy as np

from .simulation import dispersion_delay

FEATURE_NAMES = ["peak_boxcar_snr", "best_dm", "zero_dm_score", "nonzero_dm_excess",
                 "dm_peak_contrast", "best_width_samples", "channel_participation",
                 "largest_channel_fraction", "dm_score_spread", "input_max"]


def normalise(x):
    """Subtract each channel's median and divide by its robust (MAD) sigma."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite channel-by-time array")
    centred = x - np.median(x, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(centred), axis=1, keepdims=True)
    return centred / np.maximum(scale, 0.05)


class Search:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dms = np.arange(0, cfg.dm_max + cfg.dm_step / 2, cfg.dm_step)
        self.shifts = np.array([
            np.rint(dispersion_delay(dm, cfg.frequencies) / cfg.dt_s).astype(int)
            for dm in self.dms
        ])
        # only keep output samples that are valid for every trial DM (no wrap-around)
        self.nout = cfg.ntime - int(self.shifts.max())
        if self.nout < 32:
            raise ValueError("Observation too short for the configured DM search")
        self.indices = self.shifts[:, :, None] + np.arange(self.nout)[None, None, :]
        self.channels = np.arange(cfg.nchan)[None, :, None]
        self.widths = (2, 4, 8, 16)

    def transform(self, x):
        """Return the 10 features and a dict with the search details."""
        if np.shape(x) != (self.cfg.nchan, self.cfg.ntime):
            raise ValueError("Array dimensions disagree with observation configuration")
        z = normalise(x)
        aligned = z[self.channels, self.indices]           # (dm, channel, time)
        dm_time = aligned.sum(axis=1) / np.sqrt(self.cfg.nchan)

        # boxcar sums via cumulative sum, normalised by sqrt(width)
        cs = np.pad(np.cumsum(dm_time, axis=1), ((0, 0), (1, 0)))
        score_per_dm = np.full(len(self.dms), -np.inf)
        peak = -np.inf
        winner = None
        for width in self.widths:
            values = (cs[:, width:] - cs[:, :-width]) / np.sqrt(width)
            # skip the first 80 ms of every trial
            values[:, :int(0.08 / self.cfg.dt_s)] = -np.inf
            score_per_dm = np.maximum(score_per_dm, values.max(axis=1))
            row, t = np.unravel_index(np.argmax(values), values.shape)
            if values[row, t] > peak:
                peak = float(values[row, t])
                winner = (int(row), int(t), width)

        row, t, width = winner
        contribution = aligned[row, :, t:t + width].sum(axis=1) / np.sqrt(width)
        positive_part = np.maximum(contribution, 0)
        total = max(positive_part.sum(), 1e-6)
        features = np.array([
            peak, self.dms[row], score_per_dm[0], peak - score_per_dm[0],
            peak - np.median(score_per_dm), width,
            np.mean(contribution > 0.5), positive_part.max() / total,
            np.std(score_per_dm), np.max(z),
        ], dtype=np.float64)
        return features, {"dm_time": dm_time, "normalised": z, "row": row,
                          "time_index": t, "width": width,
                          "arrival_s": (t + (width - 1) / 2) * self.cfg.dt_s,
                          "dm": float(self.dms[row]), "score": peak}
