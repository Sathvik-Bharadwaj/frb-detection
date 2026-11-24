"""Make the figures used in the report, poster and slides (results/final/figures)."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from frb.detection import Search
from frb.simulation import Config, simulate

folder = root / "results/final"
figures = folder / "figures"
figures.mkdir(exist_ok=True)
summary = json.loads((folder / "summary.json").read_text())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
methods = {"classical_dm_contrast": "Dispersion contrast (baseline)",
           "physics_features_logistic": "Logistic regression (learned)"}
colors = ["#b58738", "#245c80"]
conditions = ["test_familiar", "test_unseen_interference", "test_scattered"]
labels = ["Familiar", "Sloping RFI", "Scattered", "CHIME shape"]


# 1. pipeline block diagram
fig, ax = plt.subplots(figsize=(12, 1.7))
ax.set(xlim=(0, 12), ylim=(0, 1.7))
ax.axis("off")
blocks = ["Simulate or load\na 2 s window", "Normalise\neach channel", "Dedisperse and\nboxcar search",
          "Extract 10\nfeatures", "Score and apply\nfixed threshold"]
for i, text in enumerate(blocks):
    x0 = 0.1 + i * 2.4
    ax.add_patch(FancyBboxPatch((x0, 0.15), 1.9, 1.4, boxstyle="round,pad=0.04",
                                fc="#eef3f7", ec="#2e75b6", lw=1.5))
    ax.text(x0 + 0.95, 0.85, text, ha="center", va="center", fontsize=12)
    if i < len(blocks) - 1:
        ax.annotate("", xy=(x0 + 2.36, 0.85), xytext=(x0 + 1.96, 0.85),
                    arrowprops={"arrowstyle": "->", "color": "#1f3864", "lw": 1.6})
fig.savefig(figures / "pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)


# 2. one simulated window: raw data, DM-time plane, dedispersed profile
cfg = Config()
search = Search(cfg)
x, meta = simulate(cfg, 3_000_001, True)
_, details = search.transform(x)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
axes[0].imshow(x, origin="lower", aspect="auto", cmap="magma", vmin=-2, vmax=3,
               extent=[0, cfg.ntime * cfg.dt_s, cfg.fmin_mhz, cfg.fmax_mhz])
axes[0].set(xlabel="Time (s)", ylabel="Frequency (MHz)", title="Simulated window")
axes[1].imshow(details["dm_time"], origin="lower", aspect="auto", cmap="magma",
               extent=[0, search.nout * cfg.dt_s, 0, cfg.dm_max])
axes[1].plot(meta["arrival_s"], meta["dm"], "+", color="cyan", markersize=12, mew=2,
             label="Injected pulse")
axes[1].set(xlabel="Arrival time (s)", ylabel="Trial DM (pc cm$^{-3}$)",
            title="Dedispersion search")
axes[1].legend(fontsize=9, loc="upper right")
axes[2].plot(np.arange(search.nout) * cfg.dt_s, details["dm_time"][details["row"]],
             lw=1, color="#245c80")
axes[2].axvline(meta["arrival_s"], ls="--", color="#b74c39", label="Injected arrival")
axes[2].set(xlabel="Arrival time (s)", ylabel="Summed intensity",
            title=f"Best trial, DM = {details['dm']:.0f} (true {meta['dm']:.0f})")
axes[2].legend(fontsize=9)
fig.savefig(figures / "burst_search.png", dpi=300)
plt.close(fig)


# 3. recall and false positive rate for each test condition
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
for offset, (method, label) in enumerate(methods.items()):
    rows = [summary["conditions"][condition][method] for condition in conditions]
    rows.append(summary["chime"]["semi_synthetic"][method])
    position = np.arange(4) + (offset - 0.5) * 0.34
    for axis, metric in [(axes[0], "recall"), (axes[1], "false_positive_rate")]:
        axis.bar(position, [100 * row[metric] for row in rows], width=0.32,
                 color=colors[offset], label=label)
    # small dashes = the three individual runs
    for column, metric in enumerate(["run_recall", "run_fpr"]):
        for index, row in enumerate(rows[:3]):
            axes[column].scatter(np.full(3, position[index]), 100 * np.array(row[metric]),
                                 s=40, color="#172d40", marker="_", zorder=5)
for axis in axes:
    axis.set_xticks(range(4), labels, fontsize=10.5)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
axes[0].set(ylabel="Bursts recovered (%)", ylim=(0, 100))
axes[1].set(ylabel="False positive windows (%)", ylim=(0, 2.8))
axes[1].axhline(1, linestyle="--", linewidth=1, color="#435669")
axes[1].text(3.55, 1.05, "1% target", ha="right", va="bottom", fontsize=9, color="#435669")
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc="outside upper center", ncol=2, frameon=False, fontsize=11)
fig.savefig(figures / "comparison.png", dpi=300)
fig.savefig(figures / "comparison.svg")
plt.close(fig)


# 4. recall against injected SNR
with (folder / "recovery_by_snr.csv").open() as handle:
    rows = list(csv.DictReader(handle))
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True, sharey=True)
for axis, condition, label in zip(axes, conditions, labels):
    for color, (method, method_label) in zip(colors, methods.items()):
        selected = [row for row in rows if row["condition"] == condition and row["method"] == method]
        axis.plot([float(row["snr_low"]) + 1 for row in selected],
                  [100 * float(row["recall"]) for row in selected],
                  "o-", color=color, label=method_label, markersize=4)
    axis.set(title=label, xlabel="Injected SNR", ylim=(0, 103), xticks=[5, 7, 9, 11, 13, 15])
    axis.grid(alpha=0.2)
axes[0].set_ylabel("Bursts recovered (%)")
axes[0].legend(fontsize=8.5, frameon=False, loc="lower right")
fig.savefig(figures / "recovery_snr.png", dpi=300)
plt.close(fig)


# 5. why the learned score misses CHIME-shaped bursts (needs chime_features.py)
shift_file = folder / "chime/feature_comparison.json"
if shift_file.exists():
    shift = json.loads(shift_file.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    widths = list(shift["winning_width_fraction"])
    position = np.arange(len(widths))
    for offset, (key, label, color) in enumerate([("simulated", "Simulated bursts", "#245c80"),
                                                   ("chime", "CHIME-shaped bursts", "#b74c39")]):
        values = [100 * shift["winning_width_fraction"][w][key] for w in widths]
        axes[0].bar(position + (offset - 0.5) * 0.36, values, width=0.34, color=color, label=label)
    axes[0].set_xticks(position, [f"{w} ({4 * int(w)} ms)" for w in widths])
    axes[0].set(xlabel="Winning boxcar width, samples", ylabel="Share of bursts (%)",
                title="CHIME pulse is narrower than training pulses")
    axes[0].legend(frameon=False, fontsize=10)
    features = [("channel_participation", "Channel\nparticipation"), ("dm_score_spread", "Score spread\nacross DM")]
    groups = [("median_no_pulse", "No pulse", "#9aa5b1"), ("median_chime_pulse", "CHIME-shaped burst", "#b74c39"),
              ("median_simulated_pulse", "Simulated burst", "#245c80")]
    position = np.arange(len(features))
    for offset, (key, label, color) in enumerate(groups):
        values = [shift["features"][name][key] for name, _ in features]
        axes[1].bar(position + (offset - 1) * 0.26, values, width=0.24, color=color, label=label)
    axes[1].set_xticks(position, [label for _, label in features])
    axes[1].set(ylabel="Median feature value", title="So its features look like noise")
    axes[1].legend(frameon=False, fontsize=10)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(figures / "chime_shift.png", dpi=300)
    plt.close(fig)

print(figures)
