"""Fetch ERA5 environment features around every best-track fix from Google's ARCO-ERA5 (public Zarr).

No account or API key is needed (anonymous read of the public bucket).

For each track fix (cyclone_id, time_utc, lat, lon) it writes:
  era_wmax_kt        maximum 10 m wind speed within 2.5 deg of the centre (kt)
  era_mslp_min_hpa   minimum mean-sea-level pressure within 2.5 deg of the centre (hPa)
  era_env_hpa        mean MSLP in the 5-8 deg ring around the centre (environmental pressure, hPa)
  era_dp_hpa         era_env_hpa - era_mslp_min_hpa  (ERA5 'pressure drop')
  era_grad_hpa_deg   (mean MSLP in the 2-3 deg ring - min MSLP) / 2.5 deg
  era_sst_c          sea-surface temperature within 1 deg of the centre (deg C, NaN over land)
  era_shear_ms       200-850 hPa vertical wind shear, mean over the 2-7 deg ring (m/s)
  era_steer_u_ms,    steering flow: mean of the 850, 500 and 200 hPa winds over the 2-7 deg ring (m/s),
  era_steer_v_ms     the main driver of cyclone motion

Usage
  python src/fetch_arco_era5.py                       # all fixes in data/tracks_clean.csv
  python src/fetch_arco_era5.py --stride 6            # only fixes at 00/06/12/18 UTC (half the download)
  python src/fetch_arco_era5.py --no-shear            # skip the 200/850 hPa winds (about 60% less download)
  python src/fetch_arco_era5.py --store gs://...      # use another ARCO / WeatherBench2 style store

Data volume: the default store keeps one global field per hour per variable, so every distinct hour needs
~2 MB per variable (10 fields with winds at 3 levels). Expect on the order of 100 GB of reads for the full
1982-2026 set (use --stride 6 / --no-shear to cut it, and run on a fast connection or a Google Cloud VM). Results are cached in data/era5_cache and the
script resumes where it stopped.
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
NAMES = {
    "u10": ["10m_u_component_of_wind", "u10", "10u"],
    "v10": ["10m_v_component_of_wind", "v10", "10v"],
    "msl": ["mean_sea_level_pressure", "msl"],
    "sst": ["sea_surface_temperature", "sst"],
    "u": ["u_component_of_wind", "u"],
    "v": ["v_component_of_wind", "v"],
}
# regional window that covers the whole North Indian Ocean basin (+ margin)
WIN = dict(lat=(-5.0, 45.0), lon=(35.0, 115.0))
ERA_COLS = ["era_wmax_kt", "era_mslp_min_hpa", "era_env_hpa", "era_dp_hpa",
            "era_grad_hpa_deg", "era_sst_c", "era_shear_ms", "era_steer_u_ms", "era_steer_v_ms"]
LEVELS = (200, 500, 850)


def _var(ds, key):
    for n in NAMES[key]:
        if n in ds.variables:
            return n
    raise KeyError(f"none of {NAMES[key]} found in store; available: {list(ds.variables)[:40]}")


def open_store(url):
    import xarray as xr
    kw = {"storage_options": {"token": "anon"}} if url.startswith("gs://") else {}
    return xr.open_zarr(url, chunks=None, **kw)


def _ring_mask(LA, LO, lat, lon, r0, r1):
    d = np.hypot(LA - lat, (LO - lon) * np.cos(np.radians(lat)))
    return (d >= r0) & (d <= r1), d


def patch_features(fields, lat, lon):
    """Compute the feature set for one fix. `fields` is a dict of 2-D numpy arrays on the regional window:
    keys u10, v10, msl (Pa), sst (K), optionally u200, v200, u850, v850, plus 'lat' and 'lon' 1-D axes."""
    LA, LO = np.meshgrid(fields["lat"], fields["lon"], indexing="ij")
    out = dict.fromkeys(ERA_COLS, np.nan)
    core, _ = _ring_mask(LA, LO, lat, lon, 0.0, 2.5)
    ws = np.hypot(fields["u10"], fields["v10"])
    if core.any():
        out["era_wmax_kt"] = float(np.nanmax(ws[core]) * 1.94384)
        pmin = float(np.nanmin(fields["msl"][core]) / 100.0)
        out["era_mslp_min_hpa"] = pmin
        env, _ = _ring_mask(LA, LO, lat, lon, 5.0, 8.0)
        if env.any():
            out["era_env_hpa"] = float(np.nanmean(fields["msl"][env]) / 100.0)
            out["era_dp_hpa"] = out["era_env_hpa"] - pmin
        ring, _ = _ring_mask(LA, LO, lat, lon, 2.0, 3.0)
        if ring.any():
            out["era_grad_hpa_deg"] = float((np.nanmean(fields["msl"][ring]) / 100.0 - pmin) / 2.5)
    near, _ = _ring_mask(LA, LO, lat, lon, 0.0, 1.0)
    if near.any() and np.isfinite(fields["sst"][near]).any():
        out["era_sst_c"] = float(np.nanmean(fields["sst"][near]) - 273.15)
    if "u200" in fields:
        sh, _ = _ring_mask(LA, LO, lat, lon, 2.0, 7.0)
        if sh.any():
            du = np.nanmean(fields["u200"][sh]) - np.nanmean(fields["u850"][sh])
            dv = np.nanmean(fields["v200"][sh]) - np.nanmean(fields["v850"][sh])
            out["era_shear_ms"] = float(np.hypot(du, dv))
            out["era_steer_u_ms"] = float(np.mean([np.nanmean(fields[f"u{L}"][sh]) for L in LEVELS]))
            out["era_steer_v_ms"] = float(np.mean([np.nanmean(fields[f"v{L}"][sh]) for L in LEVELS]))
    return out


def load_window(ds, t, shear=True, retries=4):
    """Read the regional window at time t for all variables -> dict of numpy arrays."""
    names = {k: _var(ds, k) for k in (["u10", "v10", "msl", "sst"] + (["u", "v"] if shear else []))}
    for attempt in range(retries):
        try:
            sel = dict(time=np.datetime64(t, "ns"), latitude=slice(WIN["lat"][1], WIN["lat"][0]),
                       longitude=slice(WIN["lon"][0], WIN["lon"][1]))
            f = {}
            for k in ("u10", "v10", "msl", "sst"):
                f[k] = ds[names[k]].sel(**sel).load().values
            a = ds[names["msl"]].sel(**sel)
            f["lat"], f["lon"] = a["latitude"].values, a["longitude"].values
            if shear:
                for k in ("u", "v"):
                    for lev in LEVELS:
                        f[f"{k}{lev}"] = ds[names[k]].sel(level=lev, **sel).load().values
            return f
        except KeyError:
            raise
        except Exception as e:                                   # transient network errors
            if attempt == retries - 1:
                print(f"  ! {t}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)


def extract(ds, fixes, shear=True, workers=8, cache_dir=None, batch=200, store_url=None):
    """fixes: DataFrame with cyclone_id, time_utc, lat, lon. Returns DataFrame of ERA5 features."""
    fixes = fixes.copy()
    fixes["time_utc"] = pd.to_datetime(fixes["time_utc"]).dt.round("1h")
    times = sorted(fixes["time_utc"].unique())
    parts = []
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    worker_local = threading.local()
    worker_stores = []
    worker_stores_lock = threading.Lock()

    def worker_ds():
        if not store_url:
            return ds
        if not hasattr(worker_local, "ds"):
            worker_local.ds = open_store(store_url)
            with worker_stores_lock:
                worker_stores.append(worker_local.ds)
        return worker_local.ds

    for b0 in range(0, len(times), batch):
        tb = times[b0:b0 + batch]
        cf = cache_dir / f"part_{b0:06d}.csv" if cache_dir else None
        if cf is not None and cf.exists():
            parts.append(pd.read_csv(cf, parse_dates=["time_utc"]))
            continue
        rows = []

        def work(t):
            f = load_window(worker_ds(), t, shear)
            sub = fixes[fixes["time_utc"] == t]
            res = []
            for r in sub.itertuples():
                feat = patch_features(f, r.lat, r.lon) if f is not None else dict.fromkeys(ERA_COLS, np.nan)
                res.append({"cyclone_id": r.cyclone_id, "time_utc": r.time_utc, **feat})
            return res

        try:
            with ThreadPoolExecutor(workers) as ex:
                for fut in as_completed([ex.submit(work, t) for t in tb]):
                    rows.extend(fut.result())
        finally:
            for worker_store in worker_stores:
                worker_store.close()
            worker_stores.clear()
            worker_local = threading.local()
        df = pd.DataFrame(rows)
        if cf is not None:
            df.to_csv(cf, index=False)
        parts.append(df)
        print(f"  {min(b0 + batch, len(times))}/{len(times)} hours done", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["cyclone_id", "time_utc"] + ERA_COLS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=str(ROOT / "data/tracks_clean.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/era5_features.csv"))
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--stride", type=int, default=3, help="keep fixes whose hour is a multiple of this (3 or 6)")
    ap.add_argument("--no-shear", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-hours", type=int, default=0, help="debug: only the first N distinct hours")
    a = ap.parse_args()

    tr = pd.read_csv(a.tracks, parse_dates=["time_utc"])
    tr = tr[tr["time_utc"].dt.hour % a.stride == 0]
    ds = open_store(a.store)
    tmax = pd.Timestamp(ds["time"].values.max())
    print(f"store time range ends {tmax}; {len(tr)} fixes, {tr['time_utc'].nunique()} distinct hours", flush=True)
    tr = tr[tr["time_utc"] <= tmax]
    if a.max_hours:
        keep = sorted(tr["time_utc"].unique())[:a.max_hours]
        tr = tr[tr["time_utc"].isin(keep)]
    res = extract(ds, tr[["cyclone_id", "time_utc", "lat", "lon"]], shear=not a.no_shear,
                  workers=a.workers, cache_dir=ROOT / "data/era5_cache", store_url=a.store)
    res.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(res)} rows; non-null wind {res['era_wmax_kt'].notna().mean():.0%}")
