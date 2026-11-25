"""Project web app: overview, live demo, results, reproducibility and code in one place.

Start it by double-clicking "Start Demo.command", or with:
    .venv/bin/streamlit run app.py
"""

import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))
from frb.detection import FEATURE_NAMES, Search
from frb.simulation import Config, simulate

st.set_page_config(page_title="Weak FRB Detection", page_icon="📡", layout="wide")

CODE_URL = "https://github.com/Sathvik-Bharadwaj/frb-detection"
TEST_SETS = {"test_familiar": "Familiar", "test_unseen_interference": "Sloping RFI (not in training)",
             "test_scattered": "Scattered pulses"}
EXAMPLES = json.loads((root / "examples/examples.json").read_text())


# ---------------------------------------------------------------- shared objects
@st.cache_resource
def detector():
    """The search, plus the saved run 1 model and its 1% false positive thresholds."""
    cfg = Config()
    folder = root / "results/final/run_1"
    metrics = json.loads((folder / "metrics.json").read_text())
    return cfg, Search(cfg), joblib.load(folder / "logistic_model.joblib"), metrics


def load_file(data, name, cfg):
    if name.lower().endswith(".npy"):
        x = np.load(io.BytesIO(data))
    else:
        text = data.decode("utf-8", errors="replace")
        x = np.loadtxt(io.StringIO(text), delimiter="," if "," in text.splitlines()[0] else None)
    x = np.asarray(x, dtype=np.float32)
    expected = (cfg.nchan, cfg.ntime)
    if x.shape == expected[::-1]:
        x = x.T
    if x.shape != expected:
        raise ValueError(f"The file must hold {expected[0]} rows (channels) x {expected[1]} columns "
                         f"(time samples). This file is {x.shape[0]} x {x.shape[1] if x.ndim > 1 else 1}.")
    if not np.isfinite(x).all():
        raise ValueError("The file contains empty, NaN or infinite values.")
    return x


def plot_search(cfg, search, candidate):
    row = int(np.argmin(np.abs(search.dms - candidate["dm"])))
    aligned = candidate["normalised"][search.channels, search.indices][row]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    axes[0].imshow(candidate["normalised"], aspect="auto", origin="lower", cmap="viridis", vmin=-3, vmax=3,
                   extent=[0, cfg.ntime * cfg.dt_s, cfg.fmin_mhz, cfg.fmax_mhz])
    axes[0].set(title="1. Input window", xlabel="Time (s)", ylabel="Frequency (MHz)")
    axes[1].imshow(candidate["dm_time"], aspect="auto", origin="lower", cmap="magma",
                   extent=[0, search.nout * cfg.dt_s, search.dms[0], search.dms[-1]])
    axes[1].plot(candidate["arrival_s"], candidate["dm"], "c+", ms=16, mew=2.5)
    axes[1].set(title="2. Dispersion search (+ = best candidate)", xlabel="Time (s)",
                ylabel="Trial DM (pc cm$^{-3}$)")
    axes[2].imshow(aligned, aspect="auto", origin="lower", cmap="viridis", vmin=-3, vmax=3,
                   extent=[0, search.nout * cfg.dt_s, cfg.fmin_mhz, cfg.fmax_mhz])
    axes[2].set(title=f"3. Delay removed at DM {search.dms[row]:.0f}", xlabel="Time (s)",
                ylabel="Frequency (MHz)")
    return fig


def show_result(x, truth):
    cfg, search, model, metrics = detector()
    thresholds = metrics["metrics"]["test_familiar"]
    features, candidate = search.transform(x)
    prob = float(model.predict_proba(features[None, :])[0, 1])
    contrast = float(features[FEATURE_NAMES.index("nonzero_dm_excess")])

    c1, c2 = st.columns(2)
    for col, name, score, thr in [
            (c1, "Learned logistic model", prob, thresholds["physics_features_logistic"]["threshold"]),
            (c2, "Dispersion contrast baseline", contrast, thresholds["classical_dm_contrast"]["threshold"])]:
        box = col.success if score > thr else col.warning
        box(f"**{name}**\n\n### {'Burst detected' if score > thr else 'No burst'}\n\n"
            f"score {score:.3f}  ·  threshold {thr:.3f}")

    found = {"DM (pc cm⁻³)": f"{candidate['dm']:.0f}",
             "Arrival time (s)": f"{candidate['arrival_s']:.3f}",
             "Peak SNR": f"{candidate['score']:.2f}"}
    table = {"Found by the search": found}
    if truth is not None:
        table["Actually in the window"] = (
            {"DM (pc cm⁻³)": f"{truth['dm']:.1f}", "Arrival time (s)": f"{truth['arrival_s']:.3f}",
             "Peak SNR": f"{truth['injected_template_snr']:.2f}"}
            if truth["label"] else {k: "no burst" for k in found})
    st.table(pd.DataFrame(table))
    st.pyplot(plot_search(cfg, search, candidate))
    st.caption("A bright burst is a curved sweep in panel 1, and panel 3 shows it straightened into a vertical "
               "line once the delay is removed. A faint burst is hidden in the noise, which is why the search is needed.")


# ---------------------------------------------------------------- page header
st.title("Robust Detection of Weak Fast Radio Bursts")
st.caption("R Sathvik Bharadwaj (2023B2A80901P) · MLEE (EEE G513) course project · Prof Ritesh Vyas")

tab_overview, tab_demo, tab_results, tab_repro, tab_code = st.tabs(
    ["Overview", "Live demo", "Results", "Reproducibility", "Code"])

# ---------------------------------------------------------------- overview
with tab_overview:
    st.markdown("""
A fast radio burst (FRB) is a millisecond flash of radio waves from far outside our galaxy.
On the way to us, the low frequencies are delayed more than the high ones (dispersion), so the
burst arrives as a faint sweep across the band instead of a sharp spike.

**How the detector works**
1. **Simulate or load** a 2-second window: 32 frequency channels from 600 to 800 MHz, 512 samples of 4 ms.
2. **Dedisperse**: for each trial dispersion measure (DM 0 to 180), shift every channel to undo the delay and add them up.
3. **Boxcar search**: slide boxes of 2, 4, 8 and 16 samples to find the brightest candidate.
4. **Features**: 10 numbers describe the candidate (peak SNR, best DM, how much brighter it is than at DM 0, ...).
5. **Decide**: a logistic regression model and a simple physical baseline (dispersion contrast) each give a score.
   Their thresholds are fixed beforehand so that only 1% of empty windows are flagged.
""")
    fig_path = root / "results/final/figures/pipeline.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Pipeline", width="stretch")

# ---------------------------------------------------------------- live demo
with tab_demo:
    cfg = detector()[0]
    mode = st.radio("Input", ["Simulate a window", "Example files", "Upload a file"], horizontal=True,
                    label_visibility="collapsed")
    x = truth = None

    if mode == "Simulate a window":
        c1, c2 = st.columns([1, 2])
        seed = c1.number_input("Seed (the same seed always gives the same window)",
                               min_value=0, max_value=2**31 - 1, value=735, step=1)
        pulse = c2.toggle("Hide a burst in the window", value=True)
        x, truth = simulate(cfg, int(seed), pulse, "familiar")

    elif mode == "Example files":
        titles = [e["title"] for e in EXAMPLES]
        choice = EXAMPLES[titles.index(st.radio("Example", titles, horizontal=True,
                                                label_visibility="collapsed"))]
        st.caption(f"{choice['description']}  (file: examples/{choice['file']})")
        x = load_file((root / "examples" / choice["file"]).read_bytes(), choice["file"], cfg)
        truth = choice["truth"]

    else:
        c1, c2 = st.columns([2, 1])
        uploaded = c1.file_uploader("A `.csv` or `.npy` file with 32 rows (channels, first row = 600 MHz) "
                                    "and 512 columns (4 ms samples)", type=["csv", "npy", "txt"])
        c2.download_button("Download an example file", (root / "examples/example_window.csv").read_bytes(),
                           file_name="example_window.csv", mime="text/csv")
        if uploaded is not None:
            try:
                x = load_file(uploaded.getvalue(), uploaded.name, cfg)
            except Exception as e:  # show any file problem as a message, not a crash
                st.error(str(e))

    st.divider()
    if x is not None:
        show_result(x, truth)
    elif mode == "Upload a file":
        st.info("Choose a file above to run the detector on it.")

# ---------------------------------------------------------------- results
with tab_results:
    summary = json.loads((root / "results/final/summary.json").read_text())
    st.subheader("Main results")
    st.markdown(f"Pooled over {summary['runs']} independent runs. Thresholds fixed on separate calibration "
                "data to give 1% false positives.")
    rows = []
    for key, label in TEST_SETS.items():
        r = summary["conditions"][key]
        rows.append({"Test set": label,
                     "Baseline recall": r["classical_dm_contrast"]["recall"],
                     "Learned recall": r["physics_features_logistic"]["recall"],
                     "Baseline false positives": r["classical_dm_contrast"]["false_positive_rate"],
                     "Learned false positives": r["physics_features_logistic"]["false_positive_rate"]})
    semi = summary["chime"]["semi_synthetic"]
    rows.append({"Test set": "Real CHIME pulse shape (FRB20180725A)",
                 "Baseline recall": semi["classical_dm_contrast"]["recall"],
                 "Learned recall": semi["physics_features_logistic"]["recall"],
                 "Baseline false positives": semi["classical_dm_contrast"]["false_positive_rate"],
                 "Learned false positives": semi["physics_features_logistic"]["false_positive_rate"]})
    st.dataframe(pd.DataFrame(rows).set_index("Test set").style.format("{:.2%}"), width="stretch")
    st.markdown("The learned model is better on pulses like its training data, including the two unseen "
                "conditions, but it falls well behind the simple baseline on the real CHIME pulse shape.")

    figures = [("comparison.png", "Recall and false positive rate by method and test set"),
               ("recovery_snr.png", "Recall against injected SNR"),
               ("burst_search.png", "Example search"),
               ("chime_shift.png", "CHIME pulse shape test")]
    cols = st.columns(2)
    for i, (name, caption) in enumerate(f for f in figures if (root / "results/final/figures" / f[0]).exists()):
        cols[i % 2].image(str(root / "results/final/figures" / name), caption=caption, width="stretch")

# ---------------------------------------------------------------- reproducibility
with tab_repro:
    st.subheader("Can the results be reproduced?")
    recorded = detector()[3]
    st.markdown("""
- **Fixed seeds.** Every simulated window comes from a numbered seed, and each data split has its own
  block of one million seeds, so training, calibration and test data never overlap.
- **Pinned versions.** `requirements-lock.txt` lists the exact version of every library.
- **Recorded provenance.** Each run's `metrics.json` stores the library versions and a SHA-256 hash of the code that produced it.
""")
    st.table(pd.DataFrame({"Recorded with the results": recorded["versions"],
                           "Running now": {"python": platform.python_version(), "numpy": np.__version__,
                                           "scikit_learn": __import__("sklearn").__version__,
                                           "scipy": __import__("scipy").__version__}}))

    st.markdown("**Re-train from scratch** (about 40 s on a laptop, up to 2 min online): rebuilds run 1 "
                "and compares it with the saved results.")
    if st.button("Re-train run 1 and compare", type="primary"):
        with st.spinner("Simulating 14,400 windows, training and evaluating..."):
            out = subprocess.run([sys.executable, str(root / "scripts/check_reproducible.py")], cwd=root,
                                 capture_output=True, text=True)
        verdict = next((line for line in out.stdout.splitlines() if line.startswith("RESULT:")),
                       "RESULT: the check did not finish.")
        (st.success if out.returncode == 0 else st.error)(verdict.removeprefix("RESULT: ").capitalize())
        st.code(out.stdout[-4000:] + out.stderr[-2000:], language="text")
    st.caption(f"This computer: {platform.system()} on {platform.machine()}. On the same kind of computer as "
               "the saved results (Apple silicon) the re-train is identical bit for bit. On a different "
               "processor every decision and count is still identical; only the last digits of some scores differ.")

# ---------------------------------------------------------------- code
with tab_code:
    st.subheader("Source code")
    st.markdown(f"The main files are below. The full project is at {CODE_URL}")
    descriptions = {
        "src/frb/simulation.py": "Simulator: noise, interference and dispersed pulses",
        "src/frb/detection.py": "Dedispersion, boxcar search and the 10 features",
        "src/frb/experiment.py": "Training, threshold calibration and evaluation",
        "scripts/check_reproducible.py": "Re-runs run 1 and compares it with the saved results",
        "app.py": "This web app",
    }
    choice = st.selectbox("File", list(descriptions), format_func=lambda f: f"{f}  —  {descriptions[f]}")
    st.code((root / choice).read_text(), language="python", line_numbers=True)
