"""Write the example windows used by the app to examples/ (each one comes from a fixed seed).

    python scripts/make_examples.py
"""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from frb.simulation import Config, simulate

# (file name, title, description, seed, contains a burst)
EXAMPLES = [
    ("example_window", "Bright burst", "A clear burst. Both methods should find it.", 2024, True),
    ("faint_burst", "Faint burst", "A weak burst (SNR 6.3). The learned model finds it; the baseline misses it.", 5434, True),
    ("noise_only", "No burst", "Noise and a short burst of interference, but no FRB. "
     "Both methods should say there is no burst.", 5002, False),
]

cfg = Config()
out = root / "examples"
out.mkdir(exist_ok=True)
index = []
for name, title, description, seed, positive in EXAMPLES:
    x, truth = simulate(cfg, seed, positive, "familiar")
    np.savetxt(out / f"{name}.csv", x, delimiter=",", fmt="%.5f")
    np.save(out / f"{name}.npy", x)
    index.append({"file": f"{name}.csv", "title": title, "description": description, "truth": truth})
(out / "examples.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"Wrote {len(index)} examples to {out}")
