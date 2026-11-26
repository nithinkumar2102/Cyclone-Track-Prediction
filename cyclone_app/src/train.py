"""Train, evaluate and save the cyclone forecast model.

  python src/train.py                 # track-only features  (works offline)
  python src/train.py --use-era5      # adds ARCO-ERA5 patch features (needs data/era5_features.csv)

Evaluation
  1. Temporal hold-out: train 1982-2018, choose models on 2019-2022, test on 2023-2026.
  2. 5-fold cross-validation grouped by cyclone (all years, every storm is predicted out-of-fold).
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import HORIZONS, ROOT
from evaluate import landfall_eval, metrics_table
from features import build_features, feature_columns, to_grid
from models import ForecastModel, baseline_predict

ERA5_STEERING_COLS = {"era_steer_u_ms", "era_steer_v_ms"}


def training_feature_columns(data, use_era5=False, steering_only=False):
    cols = feature_columns(data, use_era5)
    if steering_only:
        if not use_era5:
            raise ValueError("steering_only requires ERA5 features")
        cols = [c for c in cols if not c.startswith("era_") or c in ERA5_STEERING_COLS]
    return cols


def load_dataset(use_era5, era5_file=None):
    tracks = pd.read_csv(ROOT / "data/tracks_clean.csv", parse_dates=["time_utc"])
    era5 = None
    if use_era5:
        p = Path(era5_file) if era5_file else ROOT / "data/era5_features.csv"
        if not p.exists():
            raise SystemExit("data/era5_features.csv not found - run src/fetch_arco_era5.py first")
        era5 = pd.read_csv(p, parse_dates=["time_utc"])
    grid = to_grid(tracks)
    feats = build_features(grid, with_targets=True, era5=era5)
    feats["year"] = feats["cyclone_id"].str[:4].astype(int)
    # the 2026 record is tentative / still open: do not treat its last fix as a dissipation
    for H in HORIZONS:
        feats.loc[feats["year"] == 2026, f"y_active_{H}"] = np.nan
    base = feats[feats["observed"] & feats["lat"].notna() & (feats["wind_kt"].notna() | feats["ecp"].notna())]
    base = base.copy().reset_index(drop=True)
    return base


def concat_preds(parts):
    return {H: pd.concat([p[H] for p in parts]) for H in HORIZONS}


def run(use_era5=False, folds=5, era5_file=None, tag=None, steering_only=False):
    t0 = time.time()
    use_era5 = use_era5 or steering_only
    data = load_dataset(use_era5, era5_file)
    cols = training_feature_columns(data, use_era5, steering_only)
    tag = tag or ("era5_wind" if steering_only else "era5" if use_era5 else "track")
    out_dir = ROOT / "models" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = ROOT / "reports" / tag
    rep.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] {len(data)} base fixes, {data.cyclone_id.nunique()} cyclones, {len(cols)} features", flush=True)

    # ------------------------------------------------------------ 1) temporal hold-out
    tr, te = data[data.year <= 2022], data[data.year >= 2023]
    m = ForecastModel(cols).fit(tr)
    pred = m.predict(te)
    base = baseline_predict(m.imputer.transform(te))
    hold = metrics_table(te, pred, base)
    hold_lf = landfall_eval(te, pred, base)
    hold.to_csv(rep / "metrics_holdout_2023_2026.csv", index=False)
    print(f"hold-out done ({time.time()-t0:.0f}s): test fixes={len(te)}, cyclones={te.cyclone_id.nunique()}", flush=True)

    # ------------------------------------------------------------ 2) grouped CV (out-of-fold)
    oof_p, oof_b, oof_rows = [], [], []
    for k, (itr, ite) in enumerate(GroupKFold(n_splits=folds).split(data, groups=data["cyclone_id"])):
        trn, tst = data.iloc[itr], data.iloc[ite]
        mk = ForecastModel(cols).fit(trn)
        oof_p.append(mk.predict(tst))
        oof_b.append(baseline_predict(mk.imputer.transform(tst)))
        oof_rows.append(tst)
        print(f"  CV fold {k+1}/{folds} done ({time.time()-t0:.0f}s)", flush=True)
    cv_rows = pd.concat(oof_rows)
    cv_p, cv_b = concat_preds(oof_p), concat_preds(oof_b)
    cv = metrics_table(cv_rows, cv_p, cv_b)
    cv_lf = landfall_eval(cv_rows, cv_p, cv_b)
    cv.to_csv(rep / "metrics_cv5.csv", index=False)

    # ------------------------------------------------------------ 3) final model on all years
    full = ForecastModel(cols).fit(data)
    full.track_err_pct = {int(r.horizon_h): {"p50": float(r.track_err_median_km), "p67": float(r.track_err_p67_km),
                                             "p90": float(r.track_err_p90_km)} for r in cv.itertuples()}
    full.metrics_cv = cv.to_dict("records")
    full.metrics_holdout = hold.to_dict("records")
    full.landfall_cv = cv_lf
    full.landfall_holdout = hold_lf
    joblib.dump(full, out_dir / "model.joblib", compress=3)

    summary = {"tag": tag, "n_fixes": int(len(data)), "n_cyclones": int(data.cyclone_id.nunique()),
               "features": cols, "holdout_years": "2023-2026", "holdout_cyclones": int(te.cyclone_id.nunique()),
               "holdout_fixes": int(len(te)), "landfall_holdout": hold_lf, "landfall_cv": cv_lf,
               "chosen_models": {f"{g}_{H}": n for (g, H), n in sorted(full.choice.items())},
               "runtime_s": round(time.time() - t0)}
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[{tag}] finished in {time.time()-t0:.0f}s", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-era5", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--era5-file", default=None)
    ap.add_argument("--tag", default=None, help="output sub-folder name under models/ and reports/")
    ap.add_argument("--steering-only", action="store_true",
                    help="train using only ERA5 environmental steering u/v components")
    a = ap.parse_args()
    s = run(a.use_era5, a.folds, a.era5_file, a.tag, a.steering_only)
    print(json.dumps({k: s[k] for k in ("landfall_holdout", "landfall_cv")}, indent=2, default=float))
