"""Shared helpers: geometry, land mask, intensity classes, constants."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_data import GRADES, grade_from_wind  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [6, 12, 18, 24, 30, 36, 42, 48]          # forecast lead times (hours)
STEP_H = 3                                          # model time step (hours)
EARTH_KM = 6371.0
KT_TO_KMH = 1.852
GRADE_ORDER = ["LOPAR", "D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]
GRADE_LONG = {"LOPAR": "Low pressure area", "D": "Depression", "DD": "Deep depression",
              "CS": "Cyclonic storm", "SCS": "Severe cyclonic storm",
              "VSCS": "Very severe cyclonic storm", "ESCS": "Extremely severe cyclonic storm",
              "SuCS": "Super cyclonic storm"}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def is_land(lat, lon):
    """Vectorised land test (True over land). NaN -> False."""
    from global_land_mask import globe
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    out = np.zeros(lat.shape, dtype=bool)
    lon_w = ((lon + 180) % 360) - 180
    if ok.any():
        out[ok] = globe.is_land(np.clip(lat[ok], -89.9, 89.9), lon_w[ok])
    return out


def grade_rank(g):
    """Intensity class name -> integer rank."""
    m = {k: i for i, k in enumerate(GRADE_ORDER)}
    return np.array([m.get(x, -1) for x in g])


# ---------------------------------------------------------------------------------------------
# "Mainland" land mask: ignores small islands (e.g. Andaman/Nicobar, Lakshadweep) when deciding
# whether a cyclone has made landfall.  A land point counts as mainland when at least
# `min_frac` of the surrounding 1 degree x 1 degree box is land.
# ---------------------------------------------------------------------------------------------
_FRAC = {}
_GRID = dict(lat0=0.0, lat1=40.0, lon0=40.0, lon1=110.0, step=0.05)


def _land_fraction_grid():
    if "g" in _FRAC:
        return _FRAC["g"]
    cache = ROOT / "data" / "land_fraction_1deg.npy"
    if cache.exists():
        g = np.load(cache)
    else:
        from scipy.ndimage import uniform_filter
        la = np.arange(_GRID["lat0"], _GRID["lat1"] + 1e-9, _GRID["step"])
        lo = np.arange(_GRID["lon0"], _GRID["lon1"] + 1e-9, _GRID["step"])
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        land = is_land(LA.ravel(), LO.ravel()).reshape(LA.shape).astype(np.float32)
        g = uniform_filter(land, size=int(1.0 / _GRID["step"]), mode="nearest").astype(np.float32)
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, g)
    _FRAC["g"] = g
    return g


def land_fraction(lat, lon):
    g = _land_fraction_grid()
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    i = np.clip(np.round((lat - _GRID["lat0"]) / _GRID["step"]).astype(int), 0, g.shape[0] - 1)
    j = np.clip(np.round((lon - _GRID["lon0"]) / _GRID["step"]).astype(int), 0, g.shape[1] - 1)
    return g[i, j]


def is_mainland(lat, lon, min_frac=0.3, fill_frac=0.6):
    """Land that belongs to a large land mass (not a small island).

    Fine land points count only when the surrounding 1x1 degree box is at least `min_frac` land;
    points inside river deltas / lagoons (box >= `fill_frac` land) count as land as well.
    """
    fr = land_fraction(lat, lon)
    return (is_land(lat, lon) & (fr >= min_frac)) | (fr >= fill_frac)
