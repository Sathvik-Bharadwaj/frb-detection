"""Reader for CHIME/FRB catalogue waterfall files (HDF5)."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class BurstObservation:
    name: str
    frequencies_mhz: np.ndarray
    times_s: np.ndarray
    intensity: np.ndarray
    fitted_intensity: np.ndarray
    catalogue_dm: float


def read_chime(path):
    with h5py.File(Path(path), "r") as handle:
        burst = handle["frb"]
        frequencies = burst["plot_freq"][()]
        times = burst["plot_time"][()] / 1000
        # rows are stored high-to-low frequency, plot_freq is low-to-high
        intensity = burst["wfall"][()][::-1]
        model = burst["model_wfall"][()][::-1]
        name = burst.attrs["tns_name"]
        if isinstance(name, bytes):
            name = name.decode()
        dm = float(burst.attrs["dm"])
    if intensity.shape != (len(frequencies), len(times)):
        raise ValueError("CHIME waterfall dimensions do not match the published axes")
    if not np.all(np.diff(frequencies) > 0) or not np.all(np.diff(times) > 0):
        raise ValueError("Expected increasing frequency and time axes")
    return BurstObservation(str(name), frequencies, times, intensity, model, dm)


def fitted_profile(observation, settings):
    """Average the published model waterfall into the simulator's channels.

    Returns time relative to the profile peak, the per-channel profiles and a
    mask of channels that had any valid data.
    """
    edges = np.linspace(settings.fmin_mhz, settings.fmax_mhz, settings.nchan + 1)
    profiles = np.zeros((settings.nchan, len(observation.times_s)))
    available = np.zeros(settings.nchan, dtype=bool)
    for channel, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (observation.frequencies_mhz >= lower) & (observation.frequencies_mhz < upper)
        values = observation.fitted_intensity[selected]
        count = np.isfinite(values).sum(axis=0)
        profiles[channel] = np.divide(
            np.nansum(values, axis=0), count,
            out=np.zeros(len(observation.times_s)), where=count > 0,
        )
        available[channel] = bool(np.any(count > 0))
    if not available.any() or np.linalg.norm(profiles) == 0:
        raise ValueError("No fitted signal is available in the requested frequency band")
    peak_time = observation.times_s[np.argmax(profiles.sum(axis=0))]
    return observation.times_s - peak_time, profiles, available

