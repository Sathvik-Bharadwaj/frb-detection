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
from frb.simulation import Config, pulse_template, simulate

st.set_page_config(page_title="Weak FRB Detection", page_icon="📡", layout="wide")

CONDITIONS = {"Familiar noise and interference": "familiar",
              "Sloping interference (never seen in training)": "unseen_interference",
              "Scattered pulses (never seen in training)": "scattered"}
METHODS = {"classical_dm_contrast": "Dispersion contrast (baseline)",
           "physics_features_logistic": "Learned logistic model",
           "classical": "Peak SNR only"}
TEST_SETS = {"test_familiar": "Familiar", "test_unseen_interference": "Sloping RFI (not in training)",
             "test_scattered": "Scattered pulses"}


# ---------------------------------------------------------------- shared objects
@st.cache_resource
def detector():
    cfg = Config()
    return cfg, Search(cfg)


@st.cache_resource
def saved_run(run):
    folder = root / f"results/final/run_{run}"
    model = joblib.load(folder / "logistic_model.joblib")
    metrics = json.loads((folder / "metrics.json").read_text())
    return model, metrics["metrics"]["test_familiar"], metrics


def load_upload(uploaded, cfg):
    data = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith(".npy"):
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


def plot_overview(cfg, search, candidate):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    axes[0].imshow(candidate["normalised"], aspect="auto", origin="lower", cmap="viridis", vmin=-3, vmax=3,
                   extent=[0, cfg.ntime * cfg.dt_s, cfg.fmin_mhz, cfg.fmax_mhz])
    axes[0].set(title="1. Input: intensity over frequency and time", xlabel="Time (s)", ylabel="Frequency (MHz)")
    axes[1].imshow(candidate["dm_time"], aspect="auto", origin="lower", cmap="magma",
                   extent=[0, search.nout * cfg.dt_s, search.dms[0], search.dms[-1]])
    axes[1].plot(candidate["arrival_s"], candidate["dm"], "c+", ms=16, mew=2.5)
    axes[1].set(title="2. Dispersion search: brightness at each trial DM", xlabel="Time (s)",
                ylabel="Trial DM (pc cm$^{-3}$)")
    return fig


def plot_dedispersed(cfg, search, candidate, row):
    aligned = candidate["normalised"][search.channels, search.indices][row]
    profile = candidate["dm_time"][row]
    t = np.arange(search.nout) * cfg.dt_s
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].imshow(aligned, aspect="auto", origin="lower", cmap="viridis", vmin=-3, vmax=3,
                   extent=[0, t[-1], cfg.fmin_mhz, cfg.fmax_mhz])
    axes[0].set(ylabel="Frequency (MHz)", title=f"3. Channels shifted for DM {search.dms[row]:.0f}")
    axes[1].plot(t, profile, color="#245b7c", lw=1.2)
    axes[1].axhline(0, color="#bbb", lw=0.8)
    axes[1].set(xlabel="Time (s)", ylabel="Summed")
    return fig


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
    st.info("Open the **Live demo** tab to run the detector on an input of your choice.")

# ---------------------------------------------------------------- live demo
with tab_demo:
    cfg, search = detector()
    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.subheader("Choose the input")
        mode = st.radio("Input", ["Random simulated window", "Design your own pulse", "Upload a data file"],
                        label_visibility="collapsed")
        x = truth = None
        error = None

        if mode == "Random simulated window":
            seed = st.number_input("Seed (any whole number; the same seed always gives the same window)",
                                   min_value=0, max_value=2**31 - 1, value=735, step=1)
            pulse = st.toggle("Hide a burst in the window", value=True)
            condition = st.selectbox("Background", list(CONDITIONS))
            x, truth = simulate(cfg, int(seed), pulse, CONDITIONS[condition])

        elif mode == "Design your own pulse":
            dm = st.slider("Dispersion measure, DM (pc cm⁻³)", 0.0, 180.0, 100.0, 1.0,
                           help="How strongly the pulse is swept across the band.")
            snr = st.slider("Signal-to-noise ratio (SNR)", 2.0, 20.0, 8.0, 0.5,
                            help="Lower is fainter. Below about 6 the pulse is often missed.")
            width_ms = st.slider("Pulse width (ms)", 2.0, 30.0, 12.0, 1.0)
            arrival = st.slider("Arrival time at 800 MHz (s)", 0.2, 0.7, 0.4, 0.01)
            scatter_ms = st.slider("Scattering tail (ms)", 0.0, 65.0, 0.0, 1.0,
                                   help="Smears the pulse into a tail, stronger at low frequency.")
            seed = st.number_input("Noise seed (changes the background noise)", min_value=0,
                                   max_value=2**31 - 1, value=1, step=1)
            x, truth = simulate(cfg, int(seed), False, "familiar")
            template = pulse_template(cfg, dm, arrival, width_ms / 1000, 0.0, scatter_ms / 1000)
            x = (x + snr * template / np.linalg.norm(template)).astype(np.float32)
            truth.update(label=1, distribution="custom", dm=dm, arrival_s=arrival, width_s=width_ms / 1000,
                         injected_template_snr=snr, scatter_s=scatter_ms / 1000)

        else:
            st.markdown("A file with **32 rows** (channels, first row = 600 MHz) and **512 columns** "
                        "(4 ms samples), as `.csv` or `.npy`.")
            uploaded = st.file_uploader("Data file", type=["csv", "npy", "txt"])
            example = root / "examples/example_window.csv"
            st.download_button("Download an example file", example.read_bytes(),
                               file_name="example_window.csv", mime="text/csv")
            if uploaded is None:
                use_example = st.checkbox("Use the example file instead", value=True)
                if use_example:
                    x = np.loadtxt(example, delimiter=",").astype(np.float32)
            else:
                try:
                    x = load_upload(uploaded, cfg)
                except Exception as e:  # show any file problem as a message, not a crash
                    error = str(e)

        with st.expander("Advanced"):
            run = st.selectbox("Saved model to use", [1, 2, 3],
                               help="Each of the 3 final runs saved its own model and thresholds.")

    with right:
        if error:
            st.error(error)
        elif x is None:
            st.info("Upload a file, or tick 'Use the example file instead'.")
        else:
            model, thresholds, _ = saved_run(run)
            features, candidate = search.transform(x)
            contrast = float(features[FEATURE_NAMES.index("nonzero_dm_excess")])
            t_contrast = thresholds["classical_dm_contrast"]["threshold"]
            prob = float(model.predict_proba(features[None, :])[0, 1])
            t_prob = thresholds["physics_features_logistic"]["threshold"]

            st.subheader("Decision")
            c1, c2 = st.columns(2)
            for col, name, score, thr in [(c1, "Learned logistic model", prob, t_prob),
                                          (c2, "Dispersion contrast baseline", contrast, t_contrast)]:
                with col:
                    box = st.success if score > thr else st.warning
                    box(f"**{name}**\n\n### {'Burst detected' if score > thr else 'No burst'}\n\n"
                        f"score {score:.3f}  ·  threshold {thr:.3f}")

            found = {"DM (pc cm⁻³)": f"{candidate['dm']:.0f}",
                     "Arrival time (s)": f"{candidate['arrival_s']:.3f}",
                     "Width": f"{candidate['width'] * cfg.dt_s * 1000:.0f} ms box",
                     "Peak SNR": f"{candidate['score']:.2f}"}
            if truth is not None:
                actual = ({"DM (pc cm⁻³)": f"{truth['dm']:.1f}", "Arrival time (s)": f"{truth['arrival_s']:.3f}",
                           "Width": f"{truth['width_s'] * 1000:.1f} ms", "Peak SNR": f"{truth['injected_template_snr']:.2f} (injected)"}
                          if truth["label"] else {k: "no burst" for k in found})
                table = pd.DataFrame({"Found by the search": found, "Actually in the window": actual})
                st.table(table)
                notes = [f"Background interference: **{truth['rfi']}**"]
                if truth["label"] and truth["injected_template_snr"] < 6:
                    notes.append("SNR below 6 is very faint: about half of such bursts are missed at a 1% false alarm rate.")
                st.caption("  ·  ".join(notes))
            else:
                st.table(pd.DataFrame({"Found by the search": found}))

            st.pyplot(plot_overview(cfg, search, candidate))

            st.markdown("**Try it yourself:** move the slider. At the right DM the sweep straightens into a sharp pulse.")
            dm_choice = st.select_slider("Trial DM", options=[int(d) for d in search.dms],
                                         value=int(candidate["dm"]), key=f"dm_{mode}")
            st.pyplot(plot_dedispersed(cfg, search, candidate, int(np.where(search.dms == dm_choice)[0][0])))
            st.caption(f"The search picked DM {candidate['dm']:.0f}. The slider only changes this plot, not the decision.")

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
    df = pd.DataFrame(rows).set_index("Test set")
    st.dataframe(df.style.format("{:.2%}"), width="stretch")
    st.markdown("The learned model is better on pulses like its training data, including the two unseen "
                "conditions, but it falls well behind the simple baseline on the real CHIME pulse shape.")

    figures = [("comparison.png", "Recall and false positive rate by method and test set"),
               ("recovery_snr.png", "Recall against injected SNR"),
               ("burst_search.png", "Example search"),
               ("chime_shift.png", "CHIME pulse shape test")]
    cols = st.columns(2)
    for i, (name, caption) in enumerate(f for f in figures if (root / "results/final/figures" / f[0]).exists()):
        cols[i % 2].image(str(root / "results/final/figures" / name), caption=caption, width="stretch")
    preview = root / "results/real_preview/FRB20180725A.png"
    if preview.exists():
        with st.expander("Real CHIME burst FRB20180725A"):
            st.image(str(preview), width="stretch")
            st.caption("Source: CHIME/FRB Collaboration, Catalog 1, DOI 10.3847/1538-4365/ac33ab.")

# ---------------------------------------------------------------- reproducibility
with tab_repro:
    st.subheader("Can the results be reproduced?")
    _, _, recorded = saved_run(1)
    st.markdown("""
- **Fixed seeds.** Every simulated window comes from a numbered seed, and each data split has its own
  block of one million seeds, so training, calibration and test data never overlap.
- **Pinned versions.** `requirements-lock.txt` lists the exact version of every library.
- **Recorded provenance.** Each run's `metrics.json` stores the library versions and a SHA-256 hash of the code that produced it.
- **Checked here.** The buttons below run the checks live on this computer.
""")
    versions = pd.DataFrame({"Recorded with the results": recorded["versions"],
                             "Running now": {"python": platform.python_version(), "numpy": np.__version__,
                                             "scikit_learn": __import__("sklearn").__version__,
                                             "scipy": __import__("scipy").__version__}})
    st.table(versions)
    st.caption(f"This computer: {platform.system()} on {platform.machine()}. The saved results were made on "
               "macOS on Apple silicon (arm64). On the same kind of computer the re-train is identical bit for "
               "bit. On a different processor the last digits of floating point sums can round differently, so "
               "the check also compares every decision and count, and reports the largest numeric difference.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Unit tests** (about 35 s): delay formula, search and thresholds.")
        if st.button("Run the tests", width="stretch"):
            with st.spinner("Running tests..."):
                out = subprocess.run([sys.executable, "-m", "pytest", "-v", "--color=no"], cwd=root,
                                     capture_output=True, text=True)
            (st.success if out.returncode == 0 else st.error)(
                "All tests passed" if out.returncode == 0 else "Some tests failed")
            st.code(out.stdout[-4000:] + out.stderr[-2000:], language="text")
    with c2:
        st.markdown("**2. Re-train from scratch** (about 40 s): rebuilds run 1 and compares it with the saved results.")
        if st.button("Re-train run 1 and compare", width="stretch", type="primary"):
            with st.spinner("Simulating 14,400 windows, training and evaluating..."):
                out = subprocess.run([sys.executable, str(root / "scripts/check_reproducible.py")], cwd=root,
                                     capture_output=True, text=True)
            verdict = next((line for line in out.stdout.splitlines() if line.startswith("RESULT:")),
                           "RESULT: the check did not finish.")
            (st.success if out.returncode == 0 else st.error)(verdict.removeprefix("RESULT: ").capitalize())
            st.code(out.stdout[-4000:] + out.stderr[-2000:], language="text")

# ---------------------------------------------------------------- code
with tab_code:
    st.subheader("Source code")
    descriptions = {
        "src/frb/simulation.py": "Simulator: noise, interference and dispersed pulses",
        "src/frb/detection.py": "Normalisation, dedispersion, boxcar search and the 10 features",
        "src/frb/experiment.py": "Training, threshold calibration and evaluation of one run",
        "src/frb/observations.py": "Reader for the real CHIME waterfall",
        "scripts/run_study.py": "Runs the 3 final experiments",
        "scripts/run_experiment.py": "Runs a single experiment",
        "scripts/validate_chime.py": "CHIME profile check and transfer test",
        "scripts/summarise_study.py": "Pools the 3 runs into summary.json",
        "scripts/final_figures.py": "Makes the result figures",
        "scripts/detect.py": "Scores one chosen window from the command line",
        "scripts/check_reproducible.py": "Re-runs run 1 and compares it with the saved results",
        "tests/test_physics.py": "Tests: delay formula, search, thresholds",
        "tests/test_observations.py": "Tests: CHIME reader",
        "app.py": "This web app",
    }
    files = [f for f in descriptions if (root / f).exists()]
    choice = st.selectbox("File", files, format_func=lambda f: f"{f}  —  {descriptions[f]}")
    source = (root / choice).read_text()
    st.caption(f"{len(source.splitlines())} lines")
    st.code(source, language="python", line_numbers=True)
