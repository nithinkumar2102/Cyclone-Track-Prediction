"""Clean and merge the IMD best-track workbook (one sheet per year) into one tidy CSV.

Usage:  python src/clean_data.py --xlsx <path to workbook> --out data/tracks_clean.csv
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COLS = ["serial", "basin", "name", "date", "time", "lat", "lon", "ci",
        "ecp", "wind_kt", "dp", "grade", "oci_hpa", "oci_diam"]

# IMD intensity classes by max sustained wind (kt)
GRADES = [(0, "LOPAR"), (17, "D"), (28, "DD"), (34, "CS"), (48, "SCS"),
          (64, "VSCS"), (90, "ESCS"), (120, "SuCS")]


def grade_from_wind(w):
    """IMD category from wind (kt)."""
    w = np.asarray(w, dtype=float)
    out = np.full(w.shape, "", dtype=object)
    for thr, name in GRADES:
        out[w >= thr] = name
    out[np.isnan(w)] = ""
    return out


def _find_header(raw):
    for i in range(min(6, len(raw))):
        if any("serial" in str(x).lower() for x in raw.iloc[i].values):
            return i
    return -1


def _parse_date(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
        return pd.Timestamp(v)
    return pd.to_datetime(str(v).strip(), dayfirst=True, errors="coerce")


def _parse_hhmm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        n = int(float(str(v).strip()))
    except ValueError:
        return np.nan
    h, m = divmod(n, 100)
    if 0 <= h <= 23 and 0 <= m <= 59:
        return h + m / 60.0
    return np.nan


def _is_dt(v):
    return hasattr(v, "year") and not isinstance(v, str)


def _swap(ts):
    """Swap day and month (Excel read dd/mm as mm/dd). Returns NaT when impossible."""
    try:
        return pd.Timestamp(year=ts.year, month=ts.day, day=ts.month)
    except ValueError:
        return pd.NaT


def resolve_day_month(g):
    """Fix day/month swaps inside one system (rows in file order).

    Cells stored as real Excel dates with day<=12 are ambiguous. We pick, for each run of identical
    date cells, the reading (as-is or swapped) that keeps the sequence moving forward in time,
    and choose the start reading with the fewest violations.
    """
    dates = g["date"].tolist()
    kinds = g["date_kind"].tolist()
    # runs of consecutive identical cells
    runs = []
    for i, (d, k) in enumerate(zip(dates, kinds)):
        if runs and runs[-1]["raw"] == d:
            runs[-1]["idx"].append(i)
        else:
            runs.append({"raw": d, "kind": k, "idx": [i]})
    cands = []
    for r in runs:
        d = pd.Timestamp(r["raw"]) if not isinstance(r["raw"], str) else _parse_date(r["raw"])
        c = [d]
        if r["kind"] == "dt" and pd.notna(d) and d.day <= 12 and d.day != d.month:
            sw = _swap(d)
            if pd.notna(sw):
                c.append(sw)
        cands.append(c)
    best = None
    for start in range(len(cands[0])):
        chosen, bad = [cands[0][start]], 0
        for c in cands[1:]:
            prev = chosen[-1]
            def cost(x):
                if pd.isna(x) or pd.isna(prev):
                    return 1e9
                dd = (x - prev).days
                return dd if 0 <= dd <= 4 else 1000 + abs(dd)
            x = min(c, key=cost)
            bad += cost(x) >= 1000
            chosen.append(x)
        if best is None or bad < best[0]:
            best = (bad, chosen)
    out = [pd.NaT] * len(dates)
    for r, v in zip(runs, best[1]):
        for i in r["idx"]:
            out[i] = v
    return pd.Series(out, index=g.index)


def read_sheet(raw, sheet):
    hr = _find_header(raw)
    body = raw.iloc[hr + 1:, :14].copy()
    body.columns = COLS
    body = body.reset_index(drop=True)
    body["sheet"] = str(sheet)
    body["row_in_sheet"] = body.index + hr + 2  # excel row number
    return body


def build(xlsx):
    raw = pd.read_excel(xlsx, sheet_name=None, header=None, dtype=object)
    frames = [read_sheet(df, s) for s, df in raw.items()]
    df = pd.concat(frames, ignore_index=True)
    rep = {"rows_raw": int(len(df))}

    for c in ["lat", "lon", "ci", "ecp", "wind_kt", "dp", "oci_hpa", "oci_diam"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["serial"] = pd.to_numeric(df["serial"], errors="coerce")

    # Separator rows (no position) are dropped; date / serial / name are carried down within a sheet.
    df["date_kind"] = df["date"].map(lambda v: "dt" if _is_dt(v) else "str")
    df["date"] = df.groupby("sheet")["date"].ffill()
    df["date_kind"] = df["date_kind"].where(df["date"].notna() & df["date_kind"].notna(), "str")
    df["serial"] = df.groupby("sheet")["serial"].ffill()
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    rep["rows_with_position"] = int(len(df))

    # dates: resolve Excel day/month swaps per system, in original file order
    df["date_raw"] = df["date"]
    parts = []
    for _, g in df.groupby(["sheet", "serial"], sort=False):
        parts.append(resolve_day_month(g))
    resolved = pd.concat(parts)
    rep["dates_day_month_corrected"] = int(
        (df["date"].map(_parse_date).reindex(resolved.index) != resolved).sum())
    df["date"] = resolved.reindex(df.index)
    df["hour"] = df["time"].map(_parse_hhmm)
    df["basin"] = df["basin"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    df["basin"] = df["basin"].replace({"ARB": "AS", "ARABIAN SEA": "AS"})
    df["grade"] = df["grade"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    df["name"] = (df["name"].astype(str).str.strip().str.upper()
                  .replace({"NAN": np.nan, "-": np.nan, "": np.nan}))

    df = df[df["date"].notna() & df["hour"].notna() & df["serial"].notna()].copy()
    df["time_utc"] = df["date"].dt.normalize() + pd.to_timedelta(df["hour"], unit="h")

    # sheet year -> cyclone id (sheet names 2001/2002 are swapped in the workbook, so use the data's own year)
    df["year"] = df.groupby(["sheet", "serial"])["time_utc"].transform("min").dt.year
    df["cyclone_id"] = (df["year"].astype(int).astype(str) + "-"
                        + df["serial"].astype(int).astype(str).str.zfill(2))
    # a serial that is reused inside a sheet (rare) would merge two storms: split on big time gaps
    df = df.sort_values(["cyclone_id", "time_utc"], kind="stable")
    gap = df.groupby("cyclone_id")["time_utc"].diff().dt.total_seconds() / 3600
    df["_seg"] = (gap > 72).groupby(df["cyclone_id"]).cumsum().fillna(0).astype(int)
    df["cyclone_id"] = np.where(df["_seg"] > 0, df["cyclone_id"] + "b" + df["_seg"].astype(str), df["cyclone_id"])
    rep["gap_splits"] = int((df["_seg"] > 0).sum())

    # name / basin carried within system
    df["name"] = df.groupby("cyclone_id")["name"].transform(lambda s: s.ffill().bfill())
    df["name"] = df["name"].fillna("UNNAMED")
    df["basin"] = df.groupby("cyclone_id")["basin"].transform(lambda s: s.ffill().bfill())

    # physical plausibility (North Indian Ocean)
    bad = ~df["lat"].between(0, 40) | ~df["lon"].between(40, 110)
    rep["bad_position_rows_dropped"] = int(bad.sum())
    df = df[~bad]
    df.loc[~df["wind_kt"].between(5, 200), "wind_kt"] = np.nan
    df.loc[~df["ecp"].between(860, 1020), "ecp"] = np.nan
    df.loc[~df["dp"].between(0, 150), "dp"] = np.nan

    df = df.drop_duplicates(["cyclone_id", "time_utc"], keep="first")
    if df["grade"].isna().any():
        df["grade"] = df["grade"].fillna(pd.Series(grade_from_wind(df["wind_kt"]), index=df.index))
    df["grade_calc"] = grade_from_wind(df["wind_kt"].values)

    # hours between consecutive records
    df["dt_h"] = df.groupby("cyclone_id")["time_utc"].diff().dt.total_seconds() / 3600
    rep["rows_clean"] = int(len(df))
    rep["cyclones"] = int(df["cyclone_id"].nunique())
    rep["years"] = [int(df["year"].min()), int(df["year"].max())]
    rep["missing_pct"] = {c: round(float(df[c].isna().mean() * 100), 1)
                          for c in ["ecp", "wind_kt", "dp", "grade", "ci"]}
    rep["record_interval_hours_pct"] = (df["dt_h"].round(1).value_counts(normalize=True)
                                        .head(6).mul(100).round(1).to_dict())
    def _main_basin(s):
        s = s[~s.isin(["LAND"]) & s.notna()]
        return s.mode().iat[0] if len(s) else "UNK"
    rep["basin_counts"] = df.groupby("cyclone_id")["basin"].agg(_main_basin).value_counts().to_dict()
    rep["non_monotonic_or_big_gaps"] = int((df["dt_h"] > 24).sum())
    out = df[["cyclone_id", "year", "serial", "basin", "name", "time_utc", "lat", "lon", "ci",
              "ecp", "wind_kt", "dp", "grade", "oci_hpa", "dt_h", "sheet", "row_in_sheet"]]
    return out.reset_index(drop=True), rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", default="data/tracks_clean.csv")
    a = ap.parse_args()
    out, rep = build(a.xlsx)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    rp = Path("reports/data_quality.json")
    rp.parent.mkdir(exist_ok=True)
    rp.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
