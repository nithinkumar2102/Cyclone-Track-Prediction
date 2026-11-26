"""Accuracy metrics for the forecast model (and its baseline), plus land-crossing accuracy."""
import numpy as np
import pandas as pd

from common import HORIZONS, GRADE_ORDER, grade_from_wind, haversine_km, is_mainland
from lifecycle import land_events


def _rank(w):
    g = grade_from_wind(np.asarray(w, float))
    return np.array([GRADE_ORDER.index(x) if x in GRADE_ORDER else -1 for x in g])


def metrics_table(df, preds, base):
    """One row per lead time with model and baseline accuracy."""
    rows = []
    for H in HORIZONS:
        r = {"horizon_h": H}
        p, b = preds[H], base[H]
        # ---- track
        m = df[f"y_dlat_{H}"].notna().values
        tl, tn = df["lat"].values + df[f"y_dlat_{H}"].values, df["lon"].values + df[f"y_dlon_{H}"].values
        e = haversine_km(tl, tn, p["lat"].values, p["lon"].values)[m]
        eb = haversine_km(tl, tn, b["lat"].values, b["lon"].values)[m]
        r.update(n_track=int(m.sum()), track_err_km=e.mean(), track_err_median_km=np.median(e),
                 track_err_p67_km=np.percentile(e, 67), track_err_p90_km=np.percentile(e, 90),
                 track_within_100km_pct=100 * (e <= 100).mean(), track_within_200km_pct=100 * (e <= 200).mean(),
                 base_track_err_km=eb.mean(), track_skill_pct=100 * (1 - e.mean() / eb.mean()))
        # ---- wind
        y = df[f"y_wind_{H}"].values
        m = np.isfinite(y)
        e = np.abs(y[m] - p["wind"].values[m])
        eb = np.abs(y[m] - b["wind"].values[m])
        rk_t, rk_p = _rank(y[m]), _rank(p["wind"].values[m])
        r.update(n_wind=int(m.sum()), wind_mae_kt=e.mean(), wind_rmse_kt=np.sqrt((e ** 2).mean()),
                 wind_within_10kt_pct=100 * (e <= 10).mean(), base_wind_mae_kt=eb.mean(),
                 wind_skill_pct=100 * (1 - e.mean() / eb.mean()),
                 category_exact_pct=100 * (rk_t == rk_p).mean(),
                 category_within1_pct=100 * (np.abs(rk_t - rk_p) <= 1).mean(),
                 base_category_exact_pct=100 * (rk_t == _rank(b["wind"].values[m])).mean())
        # ---- pressure and pressure drop
        for k, nm in (("ecp", "pressure"), ("dp", "pdrop")):
            y = df[f"y_{k}_{H}"].values
            m = np.isfinite(y)
            e = np.abs(y[m] - p[k].values[m])
            eb = np.abs(y[m] - b[k].values[m])
            r.update({f"n_{nm}": int(m.sum()), f"{nm}_mae_hpa": e.mean(),
                      f"{nm}_within_5hpa_pct": 100 * (e <= 5).mean(), f"base_{nm}_mae_hpa": eb.mean(),
                      f"{nm}_skill_pct": 100 * (1 - e.mean() / eb.mean())})
        # ---- still active at +H (dissipation)
        ya = df[f"y_active_{H}"].values
        m = np.isfinite(ya)
        if m.sum() > 20 and 0 < ya[m].mean() < 1:
            from sklearn.metrics import roc_auc_score
            pa = p["p_active"].values[m]
            r.update(active_acc_pct=100 * ((pa >= 0.5) == (ya[m] == 1)).mean(),
                     active_base_acc_pct=100 * ya[m].mean(), active_auc=roc_auc_score(ya[m], pa))
        rows.append(r)
    return pd.DataFrame(rows)


def landfall_eval(df, preds, base, min_window_h=12, step_h=0.5):
    """Land-crossing accuracy: for each forecast issued while the cyclone is at sea, compare the
    first predicted mainland landfall inside the verification window with the observed one."""
    n = len(df)
    lat0, lon0 = df["lat"].values, df["lon"].values
    T = np.column_stack([lat0 + df[f"y_dlat_{H}"].values for H in HORIZONS])
    TN = np.column_stack([lon0 + df[f"y_dlon_{H}"].values for H in HORIZONS])
    out = {}
    for name, P in (("model", preds), ("baseline", base)):
        PL = np.column_stack([P[H]["lat"].values for H in HORIZONS])
        PN = np.column_stack([P[H]["lon"].values for H in HORIZONS])
        cases = hit = miss = fa = 0
        terr, perr, truth_pos = [], [], 0
        for i in range(n):
            ok = np.isfinite(T[i]) & np.isfinite(TN[i])
            k = int(np.argmin(ok)) if not ok.all() else len(HORIZONS)
            if k * 6 < min_window_h:
                continue
            t = np.r_[0, HORIZONS[:k]]
            tr = land_events(t, np.r_[lat0[i], T[i, :k]], np.r_[lon0[i], TN[i, :k]], step_h)
            if tr["starts_over_land"]:
                continue
            pr = land_events(t, np.r_[lat0[i], PL[i, :k]], np.r_[lon0[i], PN[i, :k]], step_h)
            cases += 1
            tc, pc = tr["crossings"], pr["crossings"]
            if tc:
                truth_pos += 1
            if tc and pc:
                hit += 1
                terr.append(abs(pc[0]["landfall_h"] - tc[0]["landfall_h"]))
                perr.append(haversine_km(pc[0]["landfall_lat"], pc[0]["landfall_lon"],
                                         tc[0]["landfall_lat"], tc[0]["landfall_lon"]))
            elif tc:
                miss += 1
            elif pc:
                fa += 1
        out[name] = {
            "cases": cases, "cases_with_observed_landfall": truth_pos,
            "detection_rate_pct": 100 * hit / max(hit + miss, 1),
            "false_alarm_ratio_pct": 100 * fa / max(hit + fa, 1),
            "correct_yes_no_pct": 100 * (cases - miss - fa) / max(cases, 1),
            "landfall_time_mae_h": float(np.mean(terr)) if terr else np.nan,
            "landfall_time_within_6h_pct": 100 * float(np.mean(np.array(terr) <= 6)) if terr else np.nan,
            "landfall_position_mae_km": float(np.mean(perr)) if perr else np.nan,
        }
    return out
