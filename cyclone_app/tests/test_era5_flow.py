"""Offline test of the lat/lon/date -> ERA5 -> forecast chain.

Builds a small ARCO-like Zarr store with a moving synthetic cyclone (no network), points the app at it, and checks
centre tracking and the API.   python tests/test_era5_flow.py
"""
import os
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))


def make_store(path):
    lat = np.arange(50, -10.01, -0.25)
    lon = np.arange(30, 120.01, 0.25)
    times = pd.date_range("2019-05-01 00:00", "2019-05-03 12:00", freq="h")
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    msl, u, v = (np.empty((len(times),) + LA.shape, "f4") for _ in range(3))
    truth = []
    for h in range(len(times)):
        clat, clon = 12 + 0.07 * h, 88 - 0.05 * h
        truth.append((clat, clon))
        d = np.hypot(LA - clat, (LO - clon) * np.cos(np.radians(clat)))
        msl[h] = (1010 - (20 + 0.4 * h) * np.exp(-(d / 2.0) ** 2)) * 100
        vt = (10 + 0.8 * h) * 0.514 * (d / 1.0) * np.exp(0.5 * (1 - (d / 1.0) ** 2))
        th = np.arctan2(LA - clat, LO - clon)
        u[h], v[h] = -vt * np.sin(th), vt * np.cos(th)
    ds = xr.Dataset({"mean_sea_level_pressure": (("time", "latitude", "longitude"), msl),
                     "10m_u_component_of_wind": (("time", "latitude", "longitude"), u),
                     "10m_v_component_of_wind": (("time", "latitude", "longitude"), v),
                     "u_component_of_wind": (("time", "level", "latitude", "longitude"),
                                             np.stack([u * 0.8, u, u * 1.2], axis=1)),
                     "v_component_of_wind": (("time", "level", "latitude", "longitude"),
                                             np.stack([v * 0.8, v, v * 1.2], axis=1)),
                     "sea_surface_temperature": (("time", "latitude", "longitude"), np.full(msl.shape, 302.0, "f4"))},
                    coords={"time": times, "level": [200, 500, 850], "latitude": lat, "longitude": lon}).chunk({"time": 1})
    ds.to_zarr(path, mode="w")
    return truth


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    store = os.path.join(tmp, "synthetic.zarr")
    truth = make_store(store)
    os.environ["ARCO_STORE"] = store
    from fetch_arco_era5 import open_store
    from era5_init import reconstruct
    hist, era, meta = reconstruct(open_store(store), 15.0, 85.7, "2019-05-02 12:00")
    assert len(hist) == 5, hist
    for r in hist.itertuples():
        k = int((r.time_utc - pd.Timestamp("2019-05-01")).total_seconds() // 3600)
        assert abs(r.lat - truth[k][0]) < 0.3 and abs(r.lon - truth[k][1]) < 0.3, (r, truth[k])
    unavailable = xr.Dataset(coords={"time": pd.date_range("2019-05-02", periods=1, freq="h")})
    empty_fields = {"msl": np.full((2, 2), np.nan), "u10": np.full((2, 2), np.nan),
                    "v10": np.full((2, 2), np.nan), "lat": np.array([14.0, 15.0]),
                    "lon": np.array([85.0, 86.0])}
    with patch("era5_init.load_window", return_value=empty_fields):
        try:
            reconstruct(unavailable, 15.0, 85.7, "2019-05-02 00:00")
        except RuntimeError as exc:
            assert "no data" in str(exc)
        else:
            raise AssertionError("forecast should fail when ERA5 fields contain no data")
    from fetch_arco_era5 import extract
    fixes = pd.DataFrame([
        {"cyclone_id": "SMOKE", "time_utc": pd.Timestamp("2019-05-01 00:00"), "lat": 12.0, "lon": 88.0},
        {"cyclone_id": "SMOKE", "time_utc": pd.Timestamp("2019-05-01 03:00"), "lat": 12.2, "lon": 87.9},
    ])
    local_store = open_store(store)
    try:
        extracted = extract(local_store, fixes, shear=True, workers=2, cache_dir=None,
                           batch=2, store_url=store)
    finally:
        local_store.close()
    assert np.isfinite(extracted[["era_steer_u_ms", "era_steer_v_ms"]]).all().all()
    from train import training_feature_columns
    feature_frame = pd.DataFrame(columns=[
        "cyclone_id", "time_utc", "observed", "lat", "year",
        "era_steer_u_ms", "era_steer_v_ms", "era_sst_c",
    ])
    cols = training_feature_columns(feature_frame, use_era5=True, steering_only=True)
    assert "era_steer_u_ms" in cols and "era_steer_v_ms" in cols
    assert "era_sst_c" not in cols and "year" in cols
    from server import app
    c = app.test_client()
    d = c.post("/api/forecast_era5", json={"lat": 15.0, "lon": 85.7, "time": "2019-05-02 12:00"}).json
    assert [p["h"] for p in d["forecast"]] == [6, 12, 18, 24, 30, 36, 42, 48] and len(d["era5"]["history"]) == 5
    assert c.post("/api/forecast_era5", json={"lat": 15, "lon": 85.7, "time": "2019-06-01 00:00"}).status_code == 400
    assert c.post("/api/forecast_era5", json={"lat": 30, "lon": 100, "time": "2019-05-02 12:00"}).status_code == 400
    print("ERA5 flow tests passed")
