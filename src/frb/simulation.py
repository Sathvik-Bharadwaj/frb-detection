"""Synthetic filterbank windows with optional dispersed pulses and interference."""

from dataclasses import asdict, dataclass

import numpy as np

# Cold plasma dispersion constant in s MHz^2 / (pc cm^-3)
DISPERSION_CONSTANT = 4.148808e3


@dataclass(frozen=True)
class Config:
    nchan: int = 32
    ntime: int = 512
    fmin_mhz: float = 600.0
    fmax_mhz: float = 800.0
    dt_s: float = 0.004
    dm_max: float = 180.0
    dm_step: float = 4.0

    @property
    def frequencies(self):
        # channel centre frequencies in MHz
        edges = np.linspace(self.fmin_mhz, self.fmax_mhz, self.nchan + 1)
        return (edges[:-1] + edges[1:]) / 2

    @property
    def times(self):
        return np.arange(self.ntime) * self.dt_s

    def to_dict(self):
        return asdict(self)


def dispersion_delay(dm, frequencies_mhz, reference_mhz=None):
    """Delay in seconds relative to the highest frequency (or reference_mhz)."""
    f = np.asarray(frequencies_mhz, dtype=float)
    if dm < 0 or np.any(f <= 0):
        raise ValueError("DM must be nonnegative and frequencies must be positive")
    ref = float(f.max() if reference_mhz is None else reference_mhz)
    return DISPERSION_CONSTANT * dm * (f ** -2 - ref ** -2)


def pulse_template(cfg, dm, arrival_s, width_s, spectral_index=0.0, scatter_s=0.0):
    """Dispersed Gaussian pulse, one row per channel.

    Intra-channel smearing is folded into the Gaussian width by matching the
    variance of a boxcar of length smear_s. Scattering, if requested, is a
    one-sided exponential tail that scales as f^-4.
    """
    f = cfg.frequencies
    centre = arrival_s + dispersion_delay(dm, f)
    df = (cfg.fmax_mhz - cfg.fmin_mhz) / cfg.nchan
    smear_s = 2 * DISPERSION_CONSTANT * dm * df / f ** 3
    sigma = np.sqrt(width_s ** 2 + smear_s ** 2 / 12)
    template = np.exp(-0.5 * ((cfg.times[None, :] - centre[:, None]) / sigma[:, None]) ** 2)
    template *= (f[:, None] / np.mean(f)) ** spectral_index
    if scatter_s:
        for c in range(cfg.nchan):
            tau = scatter_s * (f[c] / np.mean(f)) ** -4
            u = np.arange(min(cfg.ntime, int(np.ceil(8 * tau / cfg.dt_s)) + 1)) * cfg.dt_s
            kernel = np.exp(-u / tau)
            kernel /= kernel.sum()
            template[c] = np.convolve(template[c], kernel, mode="full")[:cfg.ntime]
    return template.astype(np.float32)


def simulate(cfg, seed, positive, distribution="familiar"):
    """Return (window, metadata). distribution is familiar, unseen_interference or scattered."""
    if distribution not in {"familiar", "unseen_interference", "scattered"}:
        raise ValueError("Unknown distribution")
    rng = np.random.default_rng(seed)

    # background: white noise with a random gain and offset per channel
    x = rng.normal(size=(cfg.nchan, cfg.ntime)).astype(np.float32)
    gain = rng.uniform(0.85, 1.15, (cfg.nchan, 1))
    x *= gain
    x += rng.uniform(-0.4, 0.4, (cfg.nchan, 1))

    # interference seen during training
    rfi_kind = rng.choice(["none", "narrowband", "impulse"], p=[0.4, 0.3, 0.3])
    if rfi_kind == "narrowband":
        c = rng.integers(0, cfg.nchan)
        x[c] += rng.uniform(2, 6)
    elif rfi_kind == "impulse":
        t0 = rng.integers(40, cfg.ntime - 40)
        x[:, t0:t0 + int(rng.integers(1, 4))] += rng.uniform(1, 3)

    # held-out test: a narrow track sloping across the band (never in training)
    if distribution == "unseen_interference":
        rfi_kind = "linear_sweep"
        centres = rng.uniform(0.1, 0.4) + np.linspace(0, rng.uniform(0.2, 0.7), cfg.nchan)
        x += rng.uniform(0.7, 1.8) * np.exp(
            -0.5 * ((cfg.times[None, :] - centres[:, None]) / 0.008) ** 2
        )

    meta = {"seed": int(seed), "label": int(positive), "distribution": distribution,
            "rfi": str(rfi_kind), "dm": None, "arrival_s": None,
            "width_s": None, "injected_template_snr": None, "scatter_s": 0.0}
    if positive:
        dm = float(rng.uniform(24, 160))
        arrival = float(rng.uniform(0.24, 0.64))
        width = float(rng.uniform(0.006, 0.022))
        snr = float(rng.uniform(4, 16))
        scatter = float(rng.uniform(0.025, 0.065)) if distribution == "scattered" else 0.0
        template = pulse_template(cfg, dm, arrival, width, rng.uniform(-2, 2), scatter)
        # scale so that the ideal matched-filter SNR equals snr
        x += snr * template / np.linalg.norm(template)
        meta.update(dm=dm, arrival_s=arrival, width_s=width,
                    injected_template_snr=snr, scatter_s=scatter)
    return x.astype(np.float32), meta
