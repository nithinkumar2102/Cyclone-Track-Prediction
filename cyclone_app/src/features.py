"""Turn best-track records into a regular 3-hourly grid and build supervised features / targets."""
import numpy as np
import pandas as pd

from common import HORIZONS, STEP_H, grade_from_wind, haversine_km, is_land, GRADE_ORDER

VARS = ["lat", "lon", "wind_kt", "ecp", "dp"]
LAGS = [3, 6, 12, 24]
ERA_COLS = ["era_wmax_kt", "era_mslp_min_hpa", "era_env_hpa", "era_dp_hpa",
            "era_grad_hpa_deg", "era_sst_c", "era_shear_ms", "era_steer_u_ms", "era_steer_v_ms"]


def _grid_one(g):
    """Regularise one cyclone to a 3-hourly grid (interpolate gaps of at most 6 h)."""
    g = g.copy()
    g["t"] = g["time_utc"].dt.round(f"{STEP_H}h")
    g = g.drop_duplicates("t").set_index("t").sort_index().drop(columns=["time_utc"])
    idx = pd.date_range(g.index.min(), g.index.max(), freq=f"{STEP_H}h")
    out = g.reindex(idx)
    out["observed"] = out["lat"].notna()
    out[VARS] = out[VARS].interpolate(limit=1, limit_area="inside")
    out["cyclone_id"] = g["cyclone_id"].iloc[0]
    out["basin"] = g["basin"].iloc[0]
    out["name"] = g["name"].iloc[0]
    out["ci"] = out["ci"]
    out.index.name = "time_utc"
    return out.reset_index()


def to_grid(tracks):
    parts = [_grid_one(g) for _, g in tracks.groupby("cyclone_id", sort=False)]
    return pd.concat(parts, ignore_index=True)


def _basin_flag(b):
    return b.map({"BOB": 1, "AS": 0}).fillna(0.5)


def build_features(grid, with_targets=True, era5=None):
    """Features at every grid row. If with_targets, also future targets for each horizon."""
    df = grid.sort_values(["cyclone_id", "time_utc"]).reset_index(drop=True).copy()
    gb = df.groupby("cyclone_id", sort=False)
    f = pd.DataFrame(index=df.index)
    f["cyclone_id"], f["time_utc"] = df["cyclone_id"], df["time_utc"]
    f["observed"] = df["observed"]
    f["lat"], f["lon"], f["wind_kt"], f["ecp"], f["dp"] = (df[c] for c in VARS)
    f["ci"] = df["ci"]
    f["age_h"] = gb.cumcount() * STEP_H
    for L in LAGS:
        k = L // STEP_H
        for v in VARS:
            f[f"d{v}_{L}h"] = df[v] - gb[v].shift(k)
    # motion vectors (km/h) from the last 6 h and 12 h
    for L in (6, 12):
        dlat, dlon = f[f"dlat_{L}h"], f[f"dlon_{L}h"]
        f[f"v_{L}h"] = dlat * 111.2 / L
        f[f"u_{L}h"] = dlon * 111.2 * np.cos(np.radians(df["lat"])) / L
        f[f"speed_{L}h"] = np.hypot(f[f"u_{L}h"], f[f"v_{L}h"])
    f["accel_u"] = f["u_6h"] - (gb["lon"].shift(2) - gb["lon"].shift(4)) * 111.2 \
        * np.cos(np.radians(df["lat"])) / 6
    f["accel_v"] = f["v_6h"] - (gb["lat"].shift(2) - gb["lat"].shift(4)) * 111.2 / 6
    doy = df["time_utc"].dt.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["basin_bob"] = _basin_flag(df["basin"])
    f["grade_rank"] = [GRADE_ORDER.index(g) if g in GRADE_ORDER else -1
                       for g in grade_from_wind(df["wind_kt"].values)]
    # land awareness: now, and where a constant-velocity track would be in 12/24/48 h
    f["land_now"] = is_land(df["lat"], df["lon"]).astype(float)
    for H in (12, 24, 48):
        la = df["lat"] + f["v_6h"].fillna(0) * H / 111.2
        lo = df["lon"] + f["u_6h"].fillna(0) * H / (111.2 * np.cos(np.radians(df["lat"])))
        f[f"land_ext_{H}h"] = is_land(la, lo).astype(float)
    if era5 is not None and len(era5):
        e = era5.copy()
        e["time_utc"] = pd.to_datetime(e["time_utc"])
        f = f.merge(e[["cyclone_id", "time_utc"] + [c for c in ERA_COLS if c in e.columns]],
                    on=["cyclone_id", "time_utc"], how="left")
        for c in ERA_COLS:
            if c not in f.columns:
                f[c] = np.nan
    if with_targets:
        for H in HORIZONS:
            k = H // STEP_H
            f[f"y_dlat_{H}"] = gb["lat"].shift(-k) - df["lat"]
            f[f"y_dlon_{H}"] = gb["lon"].shift(-k) - df["lon"]
            f[f"y_wind_{H}"] = gb["wind_kt"].shift(-k)
            f[f"y_ecp_{H}"] = gb["ecp"].shift(-k)
            f[f"y_dp_{H}"] = gb["dp"].shift(-k)
            # still a tracked system at +H?  (row exists in the cyclone's record)
            f[f"y_active_{H}"] = (gb.cumcount(ascending=False) >= k).astype(float)
    return f


def feature_columns(df, use_era5=False):
    drop = {"cyclone_id", "time_utc", "observed"}
    cols = [c for c in df.columns if c not in drop and not c.startswith("y_")]
    if not use_era5:
        cols = [c for c in cols if not c.startswith("era_")]
    return cols


def targets_for(H):
    return {"dlat": f"y_dlat_{H}", "dlon": f"y_dlon_{H}", "wind": f"y_wind_{H}",
            "ecp": f"y_ecp_{H}", "dp": f"y_dp_{H}"}


def position_error_km(lat0, lon0, dlat_true, dlon_true, dlat_pred, dlon_pred):
    return haversine_km(lat0 + dlat_true, lon0 + dlon_true, lat0 + dlat_pred, lon0 + dlon_pred)
