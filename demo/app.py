"""Local demo: python demo/app.py, then open http://127.0.0.1:8873"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from frb.simulation import Config, simulate
from frb.detection import Search

settings = Config()
search = Search(settings)
run = root / "results/final/run_1"
model = joblib.load(run / "logistic_model.joblib")
metrics = json.loads((run / "metrics.json").read_text())
threshold = metrics["metrics"]["test_familiar"]["physics_features_logistic"]["threshold"]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/":
            self.reply((root / "demo/index.html").read_bytes(), "text/html; charset=utf-8")
        elif request.path == "/chime.png":
            self.reply((root / "results/real_preview/FRB20180725A.png").read_bytes(), "image/png")
        elif request.path == "/api/example":
            try:
                parameters = parse_qs(request.query)
                seed = int(parameters.get("seed", [735])[0])
                if not 0 <= seed < 2**31:
                    raise ValueError("Seed outside supported range")
                condition = parameters.get("condition", ["familiar"])[0]
                pulse = parameters.get("pulse", ["true"])[0] == "true"
                intensity, truth = simulate(settings, seed, pulse, condition)
                features, candidate = search.transform(intensity)
                probability = float(model.predict_proba(features[None, :])[0, 1])
                result = {
                    "intensity": np.round(intensity, 3).tolist(),
                    "dm_time": np.round(candidate["dm_time"], 3).tolist(),
                    "dms": search.dms.tolist(), "dt": settings.dt_s,
                    "best_row": candidate["row"], "score": candidate["score"],
                    "probability": probability, "threshold": threshold,
                    "accepted": probability > threshold,
                    "truth": truth,
                }
                self.reply(json.dumps(result).encode(), "application/json")
            except (ValueError, KeyError) as error:
                self.reply(json.dumps({"error": str(error)}).encode(), "application/json", 400)
        else:
            self.reply(b"Not found", "text/plain", 404)

    def reply(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("Open http://127.0.0.1:8873", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8873), Handler).serve_forever()

