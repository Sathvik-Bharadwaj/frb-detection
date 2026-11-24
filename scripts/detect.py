"""Run the detector on one window chosen by the user.

Uses the saved model and thresholds from results/final/run_N (no retraining),
so the same input always gives the same output.

Examples
--------
Random simulated window (same seed -> same window, every time):
    python scripts/detect.py --seed 42
    python scripts/detect.py --seed 42 --no-pulse
    python scripts/detect.py --seed 42 --condition scattered

Pulse with parameters you choose, injected into simulated noise:
    python scripts/detect.py --dm 100 --snr 8 --width 0.01 --arrival 0.4
    python scripts/detect.py --dm 60 --snr 5 --seed 7 --scatter 0.04

Your own data (32 channels x 512 samples, channel 0 = 600 MHz, 4 ms samples):
    python scripts/detect.py --input my_window.npy
    python scripts/detect.py --input my_window.csv

A plot is saved to results/detect/ (change with --out).
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from frb.detection import FEATURE_NAMES, Search
from frb.simulation import Config, pulse_template, simulate


def load_array(path, cfg):
    path = Path(path)
    if path.suffix == ".npy":
        x = np.load(path)
    elif path.suffix in {".csv", ".txt", ".dat"}:
        x = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    else:
        raise SystemExit(f"Unsupported file type {path.suffix}: use .npy, .csv or .txt")
    x = np.asarray(x, dtype=np.float32)
    expected = (cfg.nchan, cfg.ntime)
    if x.shape == expected[::-1]:
        print(f"Note: input is {x.shape}, transposing to channels x time {expected}")
        x = x.T
    if x.shape != expected:
        raise SystemExit(f"Input must be {expected[0]} channels x {expected[1]} samples, got {x.shape}")
    if not np.isfinite(x).all():
        raise SystemExit("Input contains NaN or inf values")
    return x


def make_window(args, cfg):
    custom = args.dm is not None or args.snr is not None
    if custom:
        # background from the simulator (noise, gains, maybe RFI), then our own pulse
        x, truth = simulate(cfg, args.seed, False, "familiar")
        dm = 80.0 if args.dm is None else args.dm
        snr = 8.0 if args.snr is None else args.snr
        if not 0 <= dm <= cfg.dm_max:
            raise SystemExit(f"--dm must be between 0 and {cfg.dm_max} (the search range)")
        template = pulse_template(cfg, dm, args.arrival, args.width, args.spectral_index, args.scatter)
        x = x + snr * template / np.linalg.norm(template)
        truth.update(label=1, distribution="custom", dm=dm, arrival_s=args.arrival,
                     width_s=args.width, injected_template_snr=snr, scatter_s=args.scatter)
        return x.astype(np.float32), truth
    return simulate(cfg, args.seed, not args.no_pulse, args.condition)


def plot(path, cfg, search, x, candidate, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent_t = [0, cfg.ntime * cfg.dt_s * 1000]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    axes[0].imshow(candidate["normalised"], aspect="auto", origin="lower", cmap="viridis",
                   extent=extent_t + [cfg.fmin_mhz, cfg.fmax_mhz], vmin=-3, vmax=3)
    axes[0].set(title="Input (normalised)", xlabel="Time (ms)", ylabel="Frequency (MHz)")

    row = candidate["row"]
    aligned = candidate["normalised"][search.channels, search.indices][row]
    axes[1].imshow(aligned, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, search.nout * cfg.dt_s * 1000, cfg.fmin_mhz, cfg.fmax_mhz], vmin=-3, vmax=3)
    axes[1].set(title=f"Dedispersed at DM {candidate['dm']:.0f}", xlabel="Time (ms)")

    axes[2].imshow(candidate["dm_time"], aspect="auto", origin="lower", cmap="magma",
                   extent=[0, search.nout * cfg.dt_s * 1000, search.dms[0], search.dms[-1]])
    axes[2].set(title="DM-time plane", xlabel="Time (ms)", ylabel="Trial DM (pc cm$^{-3}$)")
    axes[2].plot(candidate["arrival_s"] * 1000, candidate["dm"], "c+", ms=14, mew=2)
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("simulated input")
    src.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    src.add_argument("--no-pulse", action="store_true", help="noise only, no pulse")
    src.add_argument("--condition", default="familiar",
                     choices=["familiar", "unseen_interference", "scattered"])
    own = p.add_argument_group("pulse you choose (any of these switches to a custom pulse)")
    own.add_argument("--dm", type=float, help="dispersion measure, 0 to 180 (default 80)")
    own.add_argument("--snr", type=float, help="injected matched-filter SNR (default 8)")
    own.add_argument("--width", type=float, default=0.012, help="pulse width in s (default 0.012)")
    own.add_argument("--arrival", type=float, default=0.4, help="arrival time at 800 MHz in s (default 0.4)")
    own.add_argument("--scatter", type=float, default=0.0, help="scattering time in s (default 0)")
    own.add_argument("--spectral-index", type=float, default=0.0)
    p.add_argument("--input", help="your own window: .npy or .csv, 32 x 512")
    p.add_argument("--run", type=int, default=1, choices=[1, 2, 3], help="which saved run's model to use")
    p.add_argument("--out", default=str(root / "results/detect"), help="folder for the plot")
    args = p.parse_args()

    cfg = Config()
    search = Search(cfg)
    run = root / f"results/final/run_{args.run}"
    model = joblib.load(run / "logistic_model.joblib")
    thresholds = json.loads((run / "metrics.json").read_text())["metrics"]["test_familiar"]

    if args.input:
        x, truth = load_array(args.input, cfg), None
        name = Path(args.input).stem
    else:
        x, truth = make_window(args, cfg)
        name = f"seed{args.seed}" + (f"_dm{truth['dm']:.0f}_snr{truth['injected_template_snr']:.1f}"
                                     if truth["label"] else "_nopulse")

    features, candidate = search.transform(x)
    contrast = features[FEATURE_NAMES.index("nonzero_dm_excess")]
    t_contrast = thresholds["classical_dm_contrast"]["threshold"]
    prob = float(model.predict_proba(features[None, :])[0, 1])
    t_prob = thresholds["physics_features_logistic"]["threshold"]

    def verdict(ok):
        return "DETECTED" if ok else "no detection"

    print()
    if truth is not None:
        print("Input (ground truth)")
        if truth["label"]:
            print(f"  pulse: DM {truth['dm']:.1f}, arrival {truth['arrival_s']:.3f} s, "
                  f"width {truth['width_s'] * 1000:.1f} ms, injected SNR {truth['injected_template_snr']:.2f}"
                  + (f", scatter {truth['scatter_s'] * 1000:.0f} ms" if truth["scatter_s"] else ""))
        else:
            print("  no pulse")
        print(f"  interference: {truth['rfi']}, condition: {truth['distribution']}, seed: {truth['seed']}")
    else:
        print(f"Input: {args.input}  shape {x.shape}")
    print("Search result")
    print(f"  best DM {candidate['dm']:.0f}, arrival {candidate['arrival_s']:.3f} s, "
          f"boxcar {candidate['width']} samples, peak SNR {candidate['score']:.2f}")
    print(f"Decisions (thresholds from run {args.run}, 1% false positive target)")
    print(f"  dispersion contrast baseline: {contrast:7.3f}  vs {t_contrast:.3f}  -> {verdict(contrast > t_contrast)}")
    print(f"  learned logistic model:       {prob:7.3f}  vs {t_prob:.3f}  -> {verdict(prob > t_prob)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure = out / f"{name}.png"
    plot(figure, cfg, search, x, candidate,
         f"{name}: contrast {verdict(contrast > t_contrast)}, learned p={prob:.3f} {verdict(prob > t_prob)}")
    print(f"Plot saved to {figure}")


if __name__ == "__main__":
    main()
