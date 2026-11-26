"""Start a forecast from just (lat, lon, date): rebuild the storm's recent history from ARCO-ERA5.

Steps
 1. read ERA5 around the given time, smooth mean-sea-level pressure and find the pressure minimum
    within `search_deg` of the point you gave (the refined cyclone centre);
 2. step back 6 h at a time (up to 24 h), each time searching near the extrapolated previous position,
    and stop when no cyclone-like low can be found (the system had not formed yet);
 3. at every centre take wind / pressure / pressure drop (and, for the ERA5-trained model, SST, shear,
    steering) from the fields;
 4. optionally convert ERA5 values to IMD-like values with data/era5_calibration.json
    (made by src/calibrate_era5.py) and/or scale to values you supply.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, haversine_km
from fetch_arco_era5 import ERA_COLS, load_window, patch_features

MIN_STRENGTH_HPA = 1.0      # (ring mean - minimum) needed to call it a low
MAX_JUMP_DEG = 3.5          # largest plausible centre move in 6 h


def _dist_deg(LA, LO, lat, lon):
    return np.hypot(LA - lat, (LO - lon) * np.cos(np.radians(lat)))


def find_center(f, guess, radius):
    """Pressure minimum of the smoothed MSLP field within `radius` degrees of `guess`."""
    from scipy.ndimage import gaussian_filter
    msl = f["msl"].astype(float) / 100.0
    sm = gaussian_filter(np.nan_to_num(msl, nan=np.nanmean(msl)), sigma=2.0)
    LA, LO = np.meshgrid(f["lat"], f["lon"], indexing="ij")
    d = _dist_deg(LA, LO, *guess)
    m = d <= radius
    if not m.any():
        return None
    idx = np.argmin(np.where(m, sm, np.inf))
    i, j = np.unravel_index(idx, sm.shape)
    clat, clon = float(LA[i, j]), float(LO[i, j])
    ring = (_dist_deg(LA, LO, clat, clon) >= 2.0) & (_dist_deg(LA, LO, clat, clon) <= 4.0)
    strength = float(np.nanmean(sm[ring]) - sm[i, j]) if ring.any() else 0.0
    return {"lat": clat, "lon": clon, "strength_hpa": strength, "on_edge": bool(d[i, j] > radius - 0.3)}


def load_calibration(root=ROOT):
    p = Path(root) / "data" / "era5_calibration.json"
    return json.loads(p.read_text()) if p.exists() else None


def reconstruct(ds, lat, lon, when, steps_back=4, step_h=6, shear=False, search_deg=2.0):
    """Returns (history DataFrame, ERA5 feature DataFrame, meta dict). History is oldest -> newest."""
    when = pd.Timestamp(when)
    tmax, tmin = pd.Timestamp(ds["time"].values.max()), pd.Timestamp(ds["time"].values.min())
    if when > tmax:
        raise ValueError(f"ARCO-ERA5 currently ends at {tmax:%Y-%m-%d %H:%M} UTC; pick a date up to then "
                         f"(for a live storm use ECMWF open data instead).")
    if when < tmin:
        raise ValueError(f"ARCO-ERA5 starts at {tmin:%Y-%m-%d}.")
    centers, rows, feats, warnings = [], [], [], []
    guess = (lat, lon)
    for k in range(steps_back + 1):
        t = when - pd.Timedelta(hours=step_h * k)
        if t < tmin:
            break
        f = load_window(ds, t, shear)
        if f is None:
            if k == 0:
                raise RuntimeError("could not read ERA5 for that time (network / store problem)")
            warnings.append(f"ERA5 unavailable at {t:%Y-%m-%d %H:%M}; history stops there")
            break
        missing = [key for key in ("msl", "u10", "v10") if not np.isfinite(f[key]).any()]
        if missing:
            detail = f"ERA5 fields have no data at {t:%Y-%m-%d %H:%M} UTC ({', '.join(missing)})"
            if k == 0:
                raise RuntimeError(detail + "; choose a date with available ERA5 observations")
            warnings.append(detail + "; history stops there")
            break
        c = find_center(f, guess, search_deg if k == 0 else 2.5)
        if c is None or c["strength_hpa"] < MIN_STRENGTH_HPA:
            if k == 0:
                raise ValueError("no cyclone-like low-pressure centre found within "
                                 f"{search_deg:.0f} degrees of that point at that time")
            warnings.append(f"no distinct low found at {t:%Y-%m-%d %H:%M}; history stops there "
                            "(system probably not formed yet)")
            break
        if centers and np.hypot(c["lat"] - centers[-1][0], c["lon"] - centers[-1][1]) > MAX_JUMP_DEG:
            warnings.append(f"centre jumped more than {MAX_JUMP_DEG} deg at {t:%Y-%m-%d %H:%M}; history stops there")
            break
        if c["on_edge"] and k == 0:
            warnings.append("the low is at the edge of the search radius - check the lat/lon you entered")
        centers.append((c["lat"], c["lon"]))
        ft = patch_features(f, c["lat"], c["lon"])
        feats.append({"cyclone_id": "LIVE", "time_utc": t, **ft})
        rows.append({"time_utc": t, "lat": c["lat"], "lon": c["lon"], "wind_kt": ft["era_wmax_kt"],
                     "ecp": ft["era_mslp_min_hpa"], "dp": ft["era_dp_hpa"], "env": ft["era_env_hpa"]})
        if len(centers) >= 2:                      # extrapolate backwards with the last displacement
            d = (centers[-2][0] - centers[-1][0], centers[-2][1] - centers[-1][1])
            guess = (centers[-1][0] - d[0], centers[-1][1] - d[1])
        else:
            guess = centers[-1]
    hist = pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)
    era = pd.DataFrame(feats).sort_values("time_utc").reset_index(drop=True)
    shift = float(haversine_km(lat, lon, rows[0]["lat"], rows[0]["lon"]))
    meta = {"center_shift_km": round(shift, 0), "n_history_points": len(hist), "warnings": warnings,
            "store_end": f"{tmax:%Y-%m-%d %H:%M}"}
    return hist, era, meta


def apply_calibration(hist, cal=None, wind_kt=None, ecp_hpa=None):
    """ERA5 -> IMD-like intensity. `cal` from calibrate_era5.py; user values rescale the whole history."""
    h = hist.copy()
    info = {"calibration": "none (raw ERA5 values - ERA5 usually under-estimates cyclone intensity)"}
    if cal:
        for col, key in (("wind_kt", "wind"), ("ecp", "ecp"), ("dp", "dp")):
            a, b = cal[key]["a"], cal[key]["b"]
            h[col] = a * h[col] + b
        info["calibration"] = f"linear ERA5->IMD fit on {cal.get('n', '?')} fixes"
    h["wind_kt"] = h["wind_kt"].clip(10, 180)
    h["ecp"] = h["ecp"].clip(880, 1016)
    h["dp"] = h["dp"].clip(0, 120)
    if wind_kt:
        r = float(wind_kt) / max(float(h["wind_kt"].iloc[-1]), 1.0)
        h["wind_kt"] = (h["wind_kt"] * r).clip(10, 180)
        info["wind_override"] = f"history wind scaled so the latest value is {float(wind_kt):.0f} kt"
    if ecp_hpa:
        off = float(ecp_hpa) - float(h["ecp"].iloc[-1])
        h["ecp"] = h["ecp"] + off
        h["dp"] = (h["dp"] - off).clip(0, 120)
        info["pressure_override"] = f"history pressure shifted so the latest value is {float(ecp_hpa):.0f} hPa"
    return h.drop(columns=["env"], errors="ignore"), info
