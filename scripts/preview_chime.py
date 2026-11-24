"""Plot the CHIME waterfall (64 sub-channels averaged) and its mean time profile."""

import argparse
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", type=Path)
parser.add_argument("--out", type=Path, default=Path("results/real_preview/FRB20180725A.png"))
args = parser.parse_args()

with h5py.File(args.path, "r") as h:
    g = h["frb"]
    x = g["wfall"][()][::-1]
    t = g["plot_time"][()]
    f = g["plot_freq"][()]
    name = g.attrs["tns_name"].decode()
    dm = float(g.attrs["dm"])
if x.shape != (len(f), len(t)):
    raise ValueError("Axes do not match the waterfall shape")
if len(f) % 64:
    raise ValueError("Expected the number of channels to be a multiple of 64")

# NaN cells are flagged data, so average with nanmean and silence empty-slice warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    reduced = np.nanmean(x.reshape(-1, 64, len(t)), axis=1)
    profile = np.nanmean(x, axis=0)
frequencies = f.reshape(-1, 64).mean(axis=1)
if frequencies[0] > frequencies[-1]:
    reduced = reduced[::-1]
    frequencies = frequencies[::-1]

dt = np.median(np.diff(t))
df = np.median(np.diff(frequencies))
extent = [t[0] - dt / 2, t[-1] + dt / 2, frequencies[0] - df / 2, frequencies[-1] + df / 2]
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#e4e4e4")

fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.8), gridspec_kw={"height_ratios": [1, 3]},
                         sharex=True, constrained_layout=True)
axes[0].plot(t, profile, color="#275b7a", lw=1.3)
axes[0].set_ylabel("Mean intensity")
axes[0].set_title(f"{name} (CHIME/FRB catalogue), DM {dm:.2f} pc cm$^{{-3}}$")
finite = reduced[np.isfinite(reduced)]
im = axes[1].imshow(np.ma.masked_invalid(reduced), origin="lower", aspect="auto", extent=extent,
                    cmap=cmap, vmin=np.percentile(finite, 2), vmax=np.percentile(finite, 98))
axes[1].set(xlabel="Time (ms)", ylabel="Frequency (MHz)")
fig.colorbar(im, ax=axes[1], label="Intensity (arb. units)")
args.out.parent.mkdir(exist_ok=True, parents=True)
fig.savefig(args.out, dpi=200)
plt.close(fig)
print(args.out)
