# Robust Detection of Weak Fast Radio Bursts Using Signal Processing and Machine Learning

R Sathvik Bharadwaj (2023B2A80901P)
MLEE (EEE G513) course project, Prof Ritesh Vyas

A dedispersion and boxcar search finds candidate pulses in a 32 channel,
600 to 800 MHz filterbank window. Ten features from the search are passed to a
logistic regression model, and the model is compared with a simple physical
baseline (dispersion contrast: best score minus the zero DM score). Both are
tested on simulated data, on interference and pulse shapes not seen in training,
and on pulses shaped like a real CHIME burst (FRB20180725A).

## Setup

Tested with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m pytest -q
```

Or run `./setup.sh`, which does the same (using `uv` if it is installed, so
Python 3.12 is fetched automatically).

## Reproducing the results

```bash
python scripts/run_study.py        # 3 runs, about 1 min each
python scripts/validate_chime.py   # CHIME profile check and transfer test
python scripts/summarise_study.py  # pools runs into results/final/summary.json
python scripts/final_figures.py    # figures in results/final/figures
```

The CHIME file is already in `data/raw`. To download it again, run
`python scripts/download_example.py`.

A quick test run with small splits:

```bash
python scripts/run_experiment.py --train 40 --calibration 40 --test 40 --seed-base 90000000 --out results/smoke
```

## Project app (demo, results, reproducibility check and code)

**Online:** https://frb-detection.streamlit.app (no installation needed; if it has
been idle it shows a "wake up" button and takes about a minute to start).

**On your own computer:**

Double-click `Start Demo.command`, or run `.venv/bin/streamlit run app.py`,
then open http://localhost:8501. The first run sets everything up
automatically. See `DEMO_GUIDE.md`.

## Running the detector on a chosen input

`scripts/detect.py` scores one window with the saved run 1 model and
thresholds (no retraining) and saves a plot to `results/detect/`.

```bash
python scripts/detect.py --seed 42                          # simulated window
python scripts/detect.py --seed 42 --no-pulse               # noise only
python scripts/detect.py --dm 100 --snr 8 --width 0.01      # pulse you choose
python scripts/detect.py --input examples/example_window.csv  # your own 32 x 512 array
```

Input files are 32 channels (row 0 = 600 MHz, row 31 = 800 MHz) by 512
samples of 4 ms, as `.npy` or `.csv`.

To check that the saved results are reproduced exactly (about 40 s):

```bash
python scripts/check_reproducible.py
```

## Demo

```bash
python demo/app.py
```

Then open http://127.0.0.1:8873. You can generate a window with or without a
pulse, see the DM search, and move the DM slider to see the pulse line up.

## Layout

| Path | Contents |
| --- | --- |
| `src/frb/simulation.py` | simulator: noise, interference, dispersed pulses |
| `src/frb/detection.py` | normalisation, dedispersion, boxcar search, features |
| `src/frb/experiment.py` | training, threshold calibration, evaluation of one run |
| `src/frb/observations.py` | CHIME waterfall reader |
| `scripts/` | study runner, CHIME check, summary, figures, packaging |
| `tests/` | unit tests for the delay formula, search and thresholds |
| `results/final/` | metrics, predictions and saved models for the 3 runs |
| `demo/` | small local web demo |

## Main results

Pooled over 3 runs (3,000 pulses and 3,000 empty windows per condition).
Thresholds are fixed on calibration data at a 1% false positive target.

| Test set | Contrast recall | Learned recall | Contrast FPR | Learned FPR |
| --- | ---: | ---: | ---: | ---: |
| Familiar | 74.53% | 84.37% | 1.07% | 1.17% |
| Sloping RFI (not in training) | 75.60% | 87.23% | 0.63% | 1.70% |
| Scattered pulses | 73.67% | 88.50% | 0.80% | 1.10% |
| CHIME pulse shape (1,200 + 1,200) | 42.08% | 15.08% | 0.50% | 0.83% |

The learned model does better on pulses that look like its training data, but
it falls well behind the baseline on the CHIME pulse shape.
