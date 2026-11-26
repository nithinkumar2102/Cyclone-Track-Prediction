"""Cyclone tracking web app (Flask).

  python app/server.py                 # http://127.0.0.1:5000   (uses models/track)
  CYCLONE_MODEL=era5 python app/server.py   # use the ERA5-enhanced model if you trained it
"""
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import Forecaster  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
_F = {}


def fc():
    if "f" not in _F:
        _F["f"] = Forecaster(os.environ.get("CYCLONE_MODEL", "track"), store=os.environ.get("ARCO_STORE"))
    return _F["f"]


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    f = fc()
    return jsonify({"ok": True, "model": f.tag, "cyclones": len(f.by_id)})


@app.get("/api/cyclones")
def cyclones():
    return jsonify(fc().catalogue())


@app.get("/api/cyclone/<cid>")
def cyclone(cid):
    if cid not in fc().by_id:
        return jsonify({"error": "unknown cyclone"}), 404
    return jsonify(fc().cyclone(cid))


@app.get("/api/forecast/<cid>")
def forecast(cid):
    if cid not in fc().by_id:
        return jsonify({"error": "unknown cyclone"}), 404
    try:
        return jsonify(fc().forecast_cyclone(cid, request.args.get("asof")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/forecast")
def forecast_live():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(fc().forecast_live(body.get("csv") or body.get("fixes") or "", body.get("name", "LIVE")))
    except Exception as e:  # bad user input
        return jsonify({"error": str(e)}), 400


@app.post("/api/forecast_era5")
def forecast_era5():
    """Body: {lat, lon, time ('YYYY-MM-DD HH:MM' UTC), name?, wind_kt?, ecp?}. Pulls ERA5 from Google ARCO."""
    b = request.get_json(silent=True) or {}
    try:
        for k in ("lat", "lon", "time"):
            if b.get(k) in (None, ""):
                raise ValueError(f"missing field: {k}")
        num = lambda v: float(v) if v not in (None, "") else None
        return jsonify(fc().forecast_from_era5(b["lat"], b["lon"], b["time"], (b.get("name") or "ERA5")[:40],
                                               num(b.get("wind_kt")), num(b.get("ecp"))))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 502
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502


@app.get("/api/accuracy")
def accuracy():
    return jsonify(fc().accuracy())


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
