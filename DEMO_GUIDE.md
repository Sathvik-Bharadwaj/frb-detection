# Presentation cheat sheet

Everything (demo, prototype, results, reproducibility check, code) is in **one app**.
No typing is needed.

## Links to give the professor

- **App (online):** https://frb-detection.streamlit.app
- **Code:** https://github.com/Sathvik-Bharadwaj/frb-detection

For the presentation itself, use the copy on your laptop (below), because it doesn't depend on Wi-Fi.
If the online app has been unused for a few days it shows a "wake up" button and takes about a minute.

## Starting the app

1. Open the `FRB_Demo` folder on the Desktop.
2. Double-click **`Start Demo.command`**.
3. A black Terminal window opens, then the app opens in your browser after about 15 seconds.
   (If it doesn't, type `localhost:8501` into the browser's address bar.)
4. **Keep the black window open** while presenting. Closing it stops the app.

Start it a couple of minutes before you present. The first start is the slowest.

If macOS says the file "cannot be opened", right-click it, choose **Open**, then **Open** again. You only need to do this once.

## The five tabs, in presenting order

| Tab | What to show | What to say |
| --- | --- | --- |
| **Overview** | Explanation and pipeline figure | The problem and the 5 steps |
| **Live demo** | The prototype. Let the professor choose the inputs. | Below |
| **Results** | Main table and figures | Learned model wins on simulated data, baseline wins on the real CHIME pulse shape |
| **Reproducibility** | Click **Re-train run 1 and compare** (about 40 s) | Shows "Identical, bit for bit": same seeds give the same numbers |
| **Code** | Pick any file from the list | The core is in `src/frb/` |

## Live demo: three ways to give an input

Pick one at the top of the Live demo tab:

- **Simulate a window:** the professor types any seed number. The same seed always gives the same window, and that is reproducibility.
  Turn off "Hide a burst in the window" to show it correctly says "No burst" on pure noise.
- **Example files:** three ready-made files: a bright burst, a faint burst, and a window with no burst.
  The faint one is the best to show: the learned model finds it and the baseline misses it.
- **Upload a file:** a `.csv` or `.npy` file with 32 rows (channels, first row = 600 MHz) and 512 columns.
  "Download an example file" gives a file in the right format.

Each time you see: the two decisions, a table of what the search found next to what was really in the window,
and three plots. In plot 1 a burst is a curved sweep; in plot 3 it is straightened into a vertical line.

## If he picks a faint pulse

From the saved results (familiar data, 1% false alarm rate):

| SNR | Baseline finds | Learned model finds |
| --- | ---: | ---: |
| 4 to 6 | 34% | 42% |
| 6 to 8 | 70% | 79% |
| 8 to 10 | 83% | 92% |
| 10 or more | about 88% | about 99% |

A miss at SNR 5 is expected, not a bug. Detecting faint bursts is the whole point of the project.
The app also says which interference was in the background (narrowband or impulse), which can hide a weak pulse.
Changing the noise seed gives a different background.

## How to explain reproducibility

- Every random number comes from a fixed seed. Each data split has its own block of one million seeds, so they never overlap.
- Exact library versions are pinned in `requirements-lock.txt`.
- Each run's `metrics.json` records the versions and a fingerprint (SHA-256 hash) of the code.
- The Reproducibility tab re-trains from scratch, live, and gets identical numbers.
- **If he asks about the online version:** it runs on a Linux server with a different processor (x86_64) from your Mac (Apple silicon).
  There, all 18,000 accept/reject decisions and every count are identical, and every recall number matches.
  Only the last digits of the scores differ, by about 1 part in 500,000.
  That's because different processors round single-precision (float32) sums slightly differently.
  On the same kind of computer it is identical bit for bit.

## If something goes wrong

- **The browser shows "connection refused":** wait 10 seconds and refresh, or double-click `Start Demo.command` again.
- **The app looks frozen after a button press:** the re-train takes about 40 seconds. Wait for it to finish.
- **Nothing works:** the slides and `results/final/figures` already show every result.

## Setting it up on another Mac

This needs the internet once. Copy the folder, double-click `Start Demo.command`, and it installs everything automatically on the first run (a few minutes).

For the command-line version, see `README.md`.
