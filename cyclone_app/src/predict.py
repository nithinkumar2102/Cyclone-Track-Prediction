"""Forecast service: model + life-cycle (start / end / land crossing / speed) for one cyclone at one time."""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import (GRADE_LONG, HORIZONS, KT_TO_KMH, ROOT, grade_from_wind, haversine_km, is_mainland)
from features import build_features, to_grid
from era5_init import apply_calibration, load_calibration, reconstruct
from fetch_arco_era5 import DEFAULT_STORE, open_store
from lifecycle import cyclone_summary, forward_speed_kmh, land_events

LIVE_COLS = ["time_utc", "lat", "lon", "wind_kt", "ecp", "dp"]


def _iso(t):
    return None if pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%MZ")


def _f(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


class Forecaster:
    def __init__(self, tag="track", root=ROOT, store=None):
        self.root = Path(root)
        self.store_url = store or DEFAULT_STORE
        self._ds = None
        self.tracks = pd.read_csv(self.root / "data/tracks_clean.csv", parse_dates=["time_utc"])
        self.model = joblib.load(self.root / "models" / tag / "model.joblib")
        self.tag = tag
        self.summary = pd.read_csv(self.root / "data/lifecycle_stats.csv",
                                   parse_dates=["start_time", "end_time", "landfall_time"])
        self.by_id = {k: g.sort_values("time_utc").reset_index(drop=True) for k, g in self.tracks.groupby("cyclone_id")}

    # ------------------------------------------------------------------ catalogue
    def catalogue(self):
        s = self.summary.copy()
        s = s[s["peak_wind_kt"].notna()].sort_values("start_time", ascending=False)
        out = []
        for r in s.itertuples():
            out.append({"id": r.cyclone_id, "name": r.name, "year": int(r.cyclone_id[:4]), "basin": r.basin,
                        "start": _iso(r.start_time), "end": _iso(r.end_time), "peak_wind_kt": _f(r.peak_wind_kt, 0),
                        "landfall": bool(r.n_landfalls > 0), "n_fixes": int(len(self.by_id[r.cyclone_id]))})
        return out

    def cyclone(self, cid):
        g = self.by_id[cid]
        row = self.summary[self.summary["cyclone_id"] == cid].iloc[0]
        return {"id": cid, "name": row["name"], "basin": row["basin"],
                "track": self._track_records(g), "lifecycle": self._observed_lifecycle(row, g)}

    def _track_records(self, g):
        spd = forward_speed_kmh(g["time_utc"], g["lat"], g["lon"])
        return [{"t": _iso(r.time_utc), "lat": _f(r.lat, 2), "lon": _f(r.lon, 2), "wind_kt": _f(r.wind_kt, 0),
                 "ecp": _f(r.ecp, 0), "dp": _f(r.dp, 0), "grade": r.grade, "speed_kmh": _f(s, 1)}
                for r, s in zip(g.itertuples(), spd)]

    def _observed_lifecycle(self, row, g):
        d = {
            "start": {"time": _iso(row["start_time"]), "lat": _f(row["start_lat"], 2), "lon": _f(row["start_lon"], 2)},
            "end": {"time": _iso(row["end_time"]), "lat": _f(row["end_lat"], 2), "lon": _f(row["end_lon"], 2),
                    "over_land": bool(row["ended_over_land"])},
            "duration_h": _f(row["duration_h"], 0), "peak_wind_kt": _f(row["peak_wind_kt"], 0),
            "min_pressure_hpa": _f(row["min_ecp_hpa"], 0), "max_pressure_drop_hpa": _f(row["max_dp_hpa"], 0),
            "mean_speed_kmh": _f(row["mean_speed_kmh"]), "max_speed_kmh": _f(row["max_speed_kmh"]),
        }
        if row["n_landfalls"] > 0:
            d["landfall"] = {"time": _iso(row["landfall_time"]), "lat": _f(row["landfall_lat"], 2),
                             "lon": _f(row["landfall_lon"], 2),
                             "hours_over_land": _f(row["hours_over_land_first"], 1),
                             "speed_kmh": _f(row["speed_at_landfall_kmh"]),
                             "dissipated_over_land": bool(row["ended_over_land"])}
        return d

    # ------------------------------------------------------------------ forecasting
    def forecast_cyclone(self, cid, asof=None):
        g = self.by_id[cid]
        asof = pd.Timestamp(asof) if asof else g["time_utc"].iloc[-1]
        hist = g[g["time_utc"] <= asof]
        if hist.empty:
            raise ValueError("as-of time is before the first fix")
        res = self._forecast(hist)
        fut = g[g["time_utc"] > hist["time_utc"].iloc[-1]]
        res["id"], res["name"] = cid, g["name"].iloc[0]
        res["actual_after"] = [{"t": _iso(r.time_utc), "lat": _f(r.lat, 2), "lon": _f(r.lon, 2),
                                "wind_kt": _f(r.wind_kt, 0), "ecp": _f(r.ecp, 0)}
                               for r in fut[fut["time_utc"] <= hist["time_utc"].iloc[-1] + pd.Timedelta(hours=48)].itertuples()]
        res["verification"] = self._verify(res, g, hist["time_utc"].iloc[-1])
        return res

    def forecast_live(self, text_or_rows, name="LIVE"):
        """Forecast from user-supplied fixes (CSV text or a list of dicts).

        columns: time_utc, lat, lon, wind_kt [, ecp, dp]   (time in UTC, e.g. 2026-11-25 06:00)
        """
        if isinstance(text_or_rows, str):
            df = pd.read_csv(io.StringIO(text_or_rows.strip()))
        else:
            df = pd.DataFrame(text_or_rows)
        df.columns = [c.strip() for c in df.columns]
        missing = {"time_utc", "lat", "lon"} - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        df["time_utc"] = pd.to_datetime(df["time_utc"])
        for c in ("wind_kt", "ecp", "dp"):
            if c not in df.columns:
                df[c] = np.nan
        df = df.sort_values("time_utc").reset_index(drop=True)
        df["cyclone_id"], df["name"] = "LIVE", name
        df["basin"] = np.where(df["lon"] >= 78, "BOB", "AS")
        df["ci"] = np.nan
        df["grade"] = grade_from_wind(df["wind_kt"].values)
        return self._forecast(df)

    # ------------------------------------------------------------------ lat / lon / date -> ERA5 -> forecast
    def _store(self):
        if self._ds is None:
            self._ds = open_store(self.store_url)
        return self._ds

    def forecast_from_era5(self, lat, lon, when, name="ERA5", wind_kt=None, ecp_hpa=None):
        """Forecast for a cyclone given only a position and a time: history and intensity come from ARCO-ERA5."""
        lat, lon = float(lat), float(lon)
        if not (-5 <= lat <= 45 and 35 <= lon <= 115):
            raise ValueError("position is outside the supported North Indian Ocean window "
                             "(lat -5..45, lon 35..115)")
        when = pd.Timestamp(when)
        when3 = when.round("3h")                          # the model works on a 3-hourly grid
        warnings = []
        if when3 != when:
            warnings.append(f"time rounded to the nearest 3 h: {when3:%Y-%m-%d %H:%M} UTC")
        needs_shear = any(c in ("era_shear_ms", "era_steer_u_ms") for c in self.model.cols)
        try:
            ds = self._store()
        except Exception as e:                            # network / store problem
            raise ConnectionError(f"could not open the ARCO-ERA5 store ({self.store_url}): {type(e).__name__}: {e}")
        hist, era, meta = reconstruct(ds, lat, lon, when3, shear=needs_shear)
        hist, cal_info = apply_calibration(hist, load_calibration(self.root), wind_kt, ecp_hpa)
        hist["cyclone_id"], hist["name"] = "LIVE", name
        hist["basin"] = np.where(hist["lon"] >= 78, "BOB", "AS")
        hist["ci"] = np.nan
        hist["grade"] = grade_from_wind(hist["wind_kt"].values)
        res = self._forecast(hist, era=era if needs_shear or any(c.startswith("era_") for c in self.model.cols) else None)
        if meta["n_history_points"] < 2:
            warnings.append("only the current position could be found, so the storm's motion is unknown - "
                            "the track forecast will be much less reliable")
        res["era5"] = {
            "requested": {"lat": lat, "lon": lon, "time": _iso(when3)},
            "center_shift_km": meta["center_shift_km"], "store_end": meta["store_end"],
            "warnings": warnings + meta["warnings"], **cal_info,
            "history": [{"t": _iso(r.time_utc), "lat": _f(r.lat, 2), "lon": _f(r.lon, 2), "wind_kt": _f(r.wind_kt, 0),
                         "ecp": _f(r.ecp, 0), "dp": _f(r.dp, 0)} for r in hist.itertuples()],
        }
        res["name"] = name
        return res

    def _forecast(self, hist, era=None):
        hist = hist.copy()
        for c in ("ci",):
            if c not in hist:
                hist[c] = np.nan
        grid = to_grid(hist)
        feats = build_features(grid, with_targets=False, era5=era)
        for c in self.model.cols:
            if c not in feats:
                feats[c] = np.nan
        last = feats[feats["observed"]].iloc[[-1]]
        pred = self.model.predict(last)
        t0 = pd.Timestamp(last["time_utc"].iloc[0])
        lat0, lon0 = float(last["lat"].iloc[0]), float(last["lon"].iloc[0])
        imp = self.model.imputer.transform(last)
        w0 = float(imp["wind_kt"].iloc[0])

        pts, prev = [], (lat0, lon0)
        for H in HORIZONS:
            p = pred[H].iloc[0]
            dist = float(haversine_km(prev[0], prev[1], p["lat"], p["lon"]))
            pts.append({"h": H, "t": _iso(t0 + pd.Timedelta(hours=H)), "lat": _f(p["lat"], 2), "lon": _f(p["lon"], 2),
                        "wind_kt": _f(p["wind"], 0), "wind_kmh": _f(p["wind"] * KT_TO_KMH, 0),
                        "pressure_hpa": _f(p["ecp"], 0), "pressure_drop_hpa": _f(p["dp"], 0),
                        "grade": str(grade_from_wind([p["wind"]])[0]), "p_active": _f(p["p_active"], 2),
                        "over_land": bool(is_mainland(p["lat"], p["lon"])),
                        "speed_kmh": _f(dist / 6.0, 1),
                        "err_radius_km": _f(self.model.track_err_pct.get(H, {}).get("p67"), 0)})
            prev = (p["lat"], p["lon"])

        # --- predicted end of the cyclone: first lead time where P(still a tracked system) < 50 %
        hh = np.array([0] + HORIZONS, float)
        pa = np.array([1.0] + [x["p_active"] if x["p_active"] is not None else 1.0 for x in pts])
        end_h = None
        for i in range(1, len(hh)):
            if pa[i] < 0.5:
                end_h = float(hh[i - 1] + (hh[i] - hh[i - 1]) * (pa[i - 1] - 0.5) / max(pa[i - 1] - pa[i], 1e-6))
                break

        # --- land crossing along the forecast path (stops at the predicted end)
        t = np.r_[0, HORIZONS].astype(float)
        la = np.r_[lat0, [x["lat"] for x in pts]]
        lo = np.r_[lon0, [x["lon"] for x in pts]]
        if end_h is not None:
            keep = t <= end_h
            t2 = np.r_[t[keep], end_h]
            la = np.interp(t2, t, la)
            lo = np.interp(t2, t, lo)
            t = t2
        ev = land_events(t, la, lo)
        landfall = None
        if ev["crossings"]:
            c = ev["crossings"][0]
            landfall = {"lead_h": _f(c["landfall_h"], 1), "time": _iso(t0 + pd.Timedelta(hours=c["landfall_h"])),
                        "lat": _f(c["landfall_lat"], 2), "lon": _f(c["landfall_lon"], 2),
                        "hours_over_land": _f(c["hours_over_land"], 1),
                        "exit_time": _iso(t0 + pd.Timedelta(hours=c["exit_h"])) if c["exit_h"] is not None else None,
                        "still_over_land_at_end": bool(c["ended_over_land"]),
                        "note": ("dissipates over land" if (c["ended_over_land"] and end_h is not None) else
                                 "still over land at the end of the 48 h window" if c["ended_over_land"] else
                                 "re-emerges over the sea")}
        spd_now = forward_speed_kmh(hist["time_utc"].tail(3), hist["lat"].tail(3), hist["lon"].tail(3))
        spd_now = float(spd_now[-1]) if len(spd_now) and np.isfinite(spd_now[-1]) else None
        first = hist.iloc[0]
        out = {
            "asof": _iso(t0), "model": self.tag,
            "current": {"lat": _f(lat0, 2), "lon": _f(lon0, 2), "wind_kt": _f(w0, 0), "wind_kmh": _f(w0 * KT_TO_KMH, 0),
                        "pressure_hpa": _f(imp["ecp"].iloc[0], 0), "pressure_drop_hpa": _f(imp["dp"].iloc[0], 0),
                        "grade": str(grade_from_wind([w0])[0]), "speed_kmh": _f(spd_now, 1),
                        "over_land": bool(is_mainland(lat0, lon0))},
            "start": {"time": _iso(first["time_utc"]), "lat": _f(first["lat"], 2), "lon": _f(first["lon"], 2)},
            "forecast": pts,
            "predicted_end": ({"lead_h": _f(end_h, 1), "time": _iso(t0 + pd.Timedelta(hours=end_h))}
                              if end_h is not None else None),
            "landfall": landfall,
            "forecast_speed": {"mean_kmh": _f(np.mean([x["speed_kmh"] for x in pts])),
                               "max_wind_kt": _f(max(x["wind_kt"] for x in pts), 0)},
            "grade_names": GRADE_LONG,
        }
        return out

    def _verify(self, res, g, t0):
        """Replay mode: compare each forecast point with what was actually observed."""
        rows = []
        gg = g.set_index("time_utc")
        for p in res["forecast"]:
            t = pd.Timestamp(p["t"].replace("Z", ""))
            if t in gg.index:
                a = gg.loc[t]
                rows.append({"h": p["h"], "track_err_km": _f(haversine_km(p["lat"], p["lon"], a["lat"], a["lon"]), 0),
                             "wind_err_kt": _f(p["wind_kt"] - a["wind_kt"], 0) if np.isfinite(a["wind_kt"]) else None,
                             "pressure_err_hpa": _f(p["pressure_hpa"] - a["ecp"], 0) if np.isfinite(a["ecp"]) else None})
        return rows

    def accuracy(self):
        m = self.model
        return {"cv": m.metrics_cv, "holdout": m.metrics_holdout,
                "landfall_cv": getattr(m, "landfall_cv", None), "landfall_holdout": getattr(m, "landfall_holdout", None),
                "choices": {f"{g}_{h}": n for (g, h), n in sorted(m.choice.items())}}
