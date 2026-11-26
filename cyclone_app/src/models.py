"""Forecast model: +6 ... +48 h position, wind, central pressure, pressure drop, and 'still active'.

Every target is learned as a *residual* on top of a simple physical baseline
 - position: constant-velocity extrapolation of the last 6 h of motion
 - intensity: persistence (value now)
For each (target, lead time) the best of {baseline, ridge, small GBM, larger GBM} is chosen on a validation
set, so the final model can never be worse than the baseline on validation data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import HORIZONS, haversine_km

REG_TARGETS = ["dlat", "dlon", "wind", "ecp", "dp"]


# --------------------------------------------------------------------------------- baselines
def base_offsets(df, H):
    """Constant-velocity extrapolation (degrees) from the last 6 h of motion."""
    v = df["v_6h"].fillna(0.0).values
    u = df["u_6h"].fillna(0.0).values
    coslat = np.cos(np.radians(df["lat"].values))
    return v * H / 111.2, u * H / (111.2 * coslat)


class IntensityImputer:
    """Fill a missing current wind from pressure, and missing pressure / pressure drop from wind
    (quadratic fits learned on the training data)."""

    def fit(self, df):
        self.coef = {}
        for c in ("ecp", "dp"):
            m = df[c].notna() & df["wind_kt"].notna()
            self.coef[c] = np.polyfit(df.loc[m, "wind_kt"], df.loc[m, c], 2)
        m = df["ecp"].notna() & df["wind_kt"].notna()
        self.coef["wind"] = np.polyfit(df.loc[m, "ecp"], df.loc[m, "wind_kt"], 2)
        return self

    def transform(self, df):
        df = df.copy()
        w_from_p = np.polyval(self.coef["wind"], df["ecp"].fillna(1005).values)
        df["wind_kt"] = df["wind_kt"].fillna(pd.Series(np.where(df["ecp"].notna(), w_from_p, 25.0), index=df.index))
        for c in ("ecp", "dp"):
            est = np.polyval(self.coef[c], df["wind_kt"].values)
            df[c] = df[c].fillna(pd.Series(est, index=df.index))
        return df


def baseline_predict(df):
    """Dict H -> DataFrame(lat, lon, wind, ecp, dp, p_active) for the baseline."""
    out = {}
    for H in HORIZONS:
        dla, dlo = base_offsets(df, H)
        out[H] = pd.DataFrame({"lat": df["lat"].values + dla, "lon": df["lon"].values + dlo,
                               "wind": df["wind_kt"].values, "ecp": df["ecp"].values,
                               "dp": df["dp"].values, "p_active": 1.0}, index=df.index)
    return out


# --------------------------------------------------------------------------------- candidates
def make_candidate(name):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=50.0))
    if name == "hgb_s":
        return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=120, max_leaf_nodes=8,
                                             min_samples_leaf=40, l2_regularization=2.0, random_state=0)
    if name == "hgb_l":
        return HistGradientBoostingRegressor(learning_rate=0.04, max_iter=250, max_leaf_nodes=16,
                                             min_samples_leaf=25, l2_regularization=1.0, random_state=0)
    if name == "hgb_l1":      # robust (absolute-error) loss: matches the MAE metric, ignores outliers
        return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=150,
                                             max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=2.0,
                                             random_state=0)
    raise ValueError(name)


CANDIDATES = ["base", "ridge", "hgb_s", "hgb_l", "hgb_l1"]
MIN_GAIN = 0.01   # a learned model must beat the baseline by >= 1 % in inner CV to be used


def _residual_target(df, H, tgt):
    if tgt == "dlat":
        return df[f"y_dlat_{H}"].values - base_offsets(df, H)[0]
    if tgt == "dlon":
        return df[f"y_dlon_{H}"].values - base_offsets(df, H)[1]
    cur = {"wind": "wind_kt", "ecp": "ecp", "dp": "dp"}[tgt]
    return df[f"y_{tgt}_{H}"].values - df[cur].values


class ForecastModel:
    def __init__(self, cols):
        self.cols = list(cols)
        self.imputer = IntensityImputer()
        self.reg = {}          # (tgt, H) -> fitted estimator or None (baseline)
        self.choice = {}       # (tgt/track, H) -> candidate name
        self.clf = {}          # H -> classifier
        self.track_err_pct = {}  # filled after evaluation: H -> {p50, p67, p90}

    # ----- fitting
    def _fit_one(self, name, X, y):
        if name == "base":
            return None
        m = make_candidate(name)
        ok = np.isfinite(y)
        m.fit(X[ok], y[ok])
        return m

    def _resid_pred(self, est, X):
        return np.zeros(len(X)) if est is None else est.predict(X)

    def fit(self, train, inner_folds=3):
        """Fit on `train`. For every (target, lead time) the best candidate is chosen by grouped
        cross-validation *inside* `train` (folds split by cyclone), then refit on all of `train`."""
        self.imputer.fit(train)
        tr = self.imputer.transform(train)
        X = tr[self.cols].values
        folds = list(GroupKFold(inner_folds).split(X, groups=tr["cyclone_id"].values))
        for H in HORIZONS:
            scores = {}
            for name in CANDIDATES:
                pr = {t: np.zeros(len(tr)) for t in REG_TARGETS}
                if name != "base":
                    for itr, ite in folds:
                        sub = tr.iloc[itr]
                        for t in REG_TARGETS:
                            est = self._fit_one(name, X[itr], _residual_target(sub, H, t))
                            pr[t][ite] = est.predict(X[ite])
                scores[name] = self._val_scores(tr, H, pr)
            for grp in ("track", "wind", "ecp", "dp"):
                best = min(CANDIDATES, key=lambda n: scores[n][grp])
                if scores[best][grp] > (1 - MIN_GAIN) * scores["base"][grp]:
                    best = "base"
                self.choice[(grp, H)] = best
            for t in REG_TARGETS:
                grp = "track" if t in ("dlat", "dlon") else t
                self.reg[(t, H)] = self._fit_one(self.choice[(grp, H)], X, _residual_target(tr, H, t))
            ya = tr[f"y_active_{H}"].values
            ok = np.isfinite(ya)
            if ok.sum() > 50 and 0 < ya[ok].mean() < 1:
                c = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100, max_leaf_nodes=8,
                                                   min_samples_leaf=40, l2_regularization=2.0, random_state=0)
                c.fit(X[ok], ya[ok].astype(int))
                self.clf[H] = c
        return self

    def _val_scores(self, va, H, pr):
        lat = va["lat"].values
        lon = va["lon"].values
        out = {}
        dla, dlo = base_offsets(va, H)
        m = va[f"y_dlat_{H}"].notna().values
        err = haversine_km(lat + va[f"y_dlat_{H}"].values, lon + va[f"y_dlon_{H}"].values,
                           lat + dla + pr["dlat"], lon + dlo + pr["dlon"])
        out["track"] = float(np.nanmean(err[m])) if m.any() else np.inf
        for t, cur in (("wind", "wind_kt"), ("ecp", "ecp"), ("dp", "dp")):
            y = va[f"y_{t}_{H}"].values
            ok = np.isfinite(y)
            out[t] = float(np.mean(np.abs(y[ok] - (va[cur].values[ok] + pr[t][ok])))) if ok.any() else np.inf
        return out

    # ----- prediction
    def predict(self, df):
        d = self.imputer.transform(df)
        for c in self.cols:                      # e.g. ERA5 columns missing at inference -> NaN
            if c not in d:
                d[c] = np.nan
        X = d[self.cols].values
        out = {}
        for H in HORIZONS:
            dla, dlo = base_offsets(d, H)
            lat = d["lat"].values + dla + self._resid_pred(self.reg[("dlat", H)], X)
            lon = d["lon"].values + dlo + self._resid_pred(self.reg[("dlon", H)], X)
            wind = d["wind_kt"].values + self._resid_pred(self.reg[("wind", H)], X)
            ecp = d["ecp"].values + self._resid_pred(self.reg[("ecp", H)], X)
            dp = d["dp"].values + self._resid_pred(self.reg[("dp", H)], X)
            pa = self.clf[H].predict_proba(X)[:, 1] if H in self.clf else np.ones(len(d))
            out[H] = pd.DataFrame({"lat": lat, "lon": lon,
                                   "wind": np.clip(wind, 10, 180),
                                   "ecp": np.clip(ecp, 880, 1016),
                                   "dp": np.clip(dp, 0, 120), "p_active": pa}, index=df.index)
        return out
