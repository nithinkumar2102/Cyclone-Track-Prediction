"""Cyclone life-cycle analysis: genesis / end, land crossing (landfall, time over land) and forward speed."""
import numpy as np
import pandas as pd

from common import haversine_km, is_land, is_mainland, KT_TO_KMH


def forward_speed_kmh(times, lat, lon):
    """Translational speed between consecutive fixes (km/h). First value is NaN."""
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    dt = t.diff().dt.total_seconds().values / 3600.0
    d = np.r_[np.nan, haversine_km(np.asarray(lat)[:-1], np.asarray(lon)[:-1],
                                   np.asarray(lat)[1:], np.asarray(lon)[1:])]
    with np.errstate(divide="ignore", invalid="ignore"):
        return d / dt


def land_events(t_hours, lat, lon, step_h=0.25, mainland=True):
    """Land crossings along a path.

    t_hours : hours since the first point, increasing.  The path is interpolated every `step_h`
    hours and tested against a land mask.  Returns a dict with every sea->land crossing.
    """
    t = np.asarray(t_hours, float)
    la, lo = np.asarray(lat, float), np.asarray(lon, float)
    ok = np.isfinite(t) & np.isfinite(la) & np.isfinite(lo)
    t, la, lo = t[ok], la[ok], lo[ok]
    if len(t) < 2:
        return {"starts_over_land": bool(len(t) and is_mainland(la[:1], lo[:1])[0]), "crossings": []}
    th = np.arange(t[0], t[-1] + 1e-9, step_h)
    pla, plo = np.interp(th, t, la), np.interp(th, t, lo)
    land = is_mainland(pla, plo) if mainland else is_land(pla, plo)
    if mainland and land.any():                      # bridge sea gaps shorter than 3 h (channels, bays)
        from scipy.ndimage import binary_closing
        k = max(int(round(3.0 / step_h)), 1)
        land = binary_closing(np.r_[False, land, False], structure=np.ones(k, bool))[1:-1] | land
    ent = np.where(~land[:-1] & land[1:])[0] + 1          # first land sample of each crossing
    ext = np.where(land[:-1] & ~land[1:])[0] + 1          # first sea sample after land
    out = []
    for e in ent:
        later = ext[ext > e]
        x = later[0] if len(later) else None
        out.append({
            "landfall_h": float(th[e]), "landfall_lat": float(pla[e]), "landfall_lon": float(plo[e]),
            "exit_h": float(th[x]) if x is not None else None,
            "hours_over_land": float((th[x] if x is not None else th[-1]) - th[e]),
            "ended_over_land": x is None,
        })
    return {"starts_over_land": bool(land[0]), "crossings": out}


def cyclone_summary(g):
    """Full life-cycle summary of one observed cyclone (g = its rows from tracks_clean)."""
    g = g.sort_values("time_utc").reset_index(drop=True)
    t0 = g["time_utc"].iloc[0]
    th = (g["time_utc"] - t0).dt.total_seconds().values / 3600.0
    ev = land_events(th, g["lat"].values, g["lon"].values)
    spd = forward_speed_kmh(g["time_utc"], g["lat"], g["lon"])
    # speed around the first landfall (mean of fixes within +-6 h)
    first = ev["crossings"][0] if ev["crossings"] else None
    sp_land = np.nan
    if first is not None:
        m = np.abs(th - first["landfall_h"]) <= 6
        sp_land = float(np.nanmean(spd[m])) if np.isfinite(spd[m]).any() else np.nan
    peak = g["wind_kt"].max()
    row = {
        "cyclone_id": g["cyclone_id"].iloc[0], "name": g["name"].iloc[0],
        "basin": g["basin"].mode().iat[0] if g["basin"].notna().any() else None,
        "start_time": t0, "start_lat": g["lat"].iloc[0], "start_lon": g["lon"].iloc[0],
        "end_time": g["time_utc"].iloc[-1], "end_lat": g["lat"].iloc[-1], "end_lon": g["lon"].iloc[-1],
        "duration_h": float(th[-1]), "peak_wind_kt": peak,
        "min_ecp_hpa": g["ecp"].min(),
        "max_dp_hpa": g["dp"].max(),
        "mean_speed_kmh": float(np.nanmean(spd)) if np.isfinite(spd).any() else np.nan,
        "max_speed_kmh": float(np.nanmax(spd)) if np.isfinite(spd).any() else np.nan,
        "n_landfalls": len(ev["crossings"]),
        "landfall_time": (t0 + pd.to_timedelta(first["landfall_h"], unit="h")) if first else pd.NaT,
        "landfall_lat": first["landfall_lat"] if first else np.nan,
        "landfall_lon": first["landfall_lon"] if first else np.nan,
        "hours_over_land_first": first["hours_over_land"] if first else np.nan,
        "ended_over_land": bool(first["ended_over_land"]) if first else False,
        "speed_at_landfall_kmh": sp_land,
        "hours_over_land_total": float(sum(c["hours_over_land"] for c in ev["crossings"])),
    }
    return row


def summarize_all(tracks):
    rows = [cyclone_summary(g) for _, g in tracks.groupby("cyclone_id")]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from common import ROOT
    d = pd.read_csv(ROOT / "data/tracks_clean.csv", parse_dates=["time_utc"])
    s = summarize_all(d)
    s.to_csv(ROOT / "data/lifecycle_stats.csv", index=False)
    print(f"wrote data/lifecycle_stats.csv ({len(s)} cyclones)")
