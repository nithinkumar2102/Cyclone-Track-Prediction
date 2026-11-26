"""Fit linear ERA5 -> IMD maps (wind, central pressure, pressure drop) from data/era5_features.csv.

  python src/calibrate_era5.py                 # writes data/era5_calibration.json
The app (ERA5 lat/lon/date tab) then converts raw ERA5 intensity into IMD-like values automatically.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    a, b = np.polyfit(x[m], y[m], 1)
    pred = a * x[m] + b
    r2 = 1 - np.sum((y[m] - pred) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
    return {"a": float(a), "b": float(b), "r2": float(r2), "n": int(m.sum()),
            "mae_before": float(np.mean(np.abs(y[m] - x[m]))), "mae_after": float(np.mean(np.abs(y[m] - pred)))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5", default=str(ROOT / "data/era5_features.csv"))
    ap.add_argument("--tracks", default=str(ROOT / "data/tracks_clean.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/era5_calibration.json"))
    a = ap.parse_args()
    e = pd.read_csv(a.era5, parse_dates=["time_utc"])
    t = pd.read_csv(a.tracks, parse_dates=["time_utc"])
    t["time_utc"] = t["time_utc"].dt.round("1h")
    d = t.merge(e, on=["cyclone_id", "time_utc"], how="inner")
    cal = {"wind": fit(d["era_wmax_kt"].values, d["wind_kt"].values),
           "ecp": fit(d["era_mslp_min_hpa"].values, d["ecp"].values),
           "dp": fit(d["era_dp_hpa"].values, d["dp"].values)}
    cal["n"] = int(len(d))
    Path(a.out).write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))
