# Cyclone Tracker – North Indian Ocean

Machine-learning forecast of cyclone **position, wind speed, central pressure (hPa) and pressure drop** at
**+6, +12, +18, +24, +30, +36, +42 and +48 h**, plus each cyclone's **start, end, land crossing (time over land)
and forward speed**, with a web app. Trained on the IMD best-track workbook (1982-2026); optionally enriched with
ERA5 fields read from **Google's ARCO-ERA5** (no CDS account / key).

```
src/clean_data.py        45 yearly sheets -> data/tracks_clean.csv (+ reports/data_quality.json)
src/lifecycle.py         start / end / land crossing / time over land / forward speed  -> data/lifecycle_stats.csv
src/fetch_arco_era5.py   ERA5 patch features around every fix from Google ARCO-ERA5   -> data/era5_features.csv
src/features.py          3-hourly grid, lags, motion, land awareness, (ERA5) features, +6..+48 h targets
src/models.py            baseline + ridge + gradient boosting per target and lead time, chosen by grouped CV
src/train.py             hold-out + 5-fold CV evaluation, saves models/<tag>/model.joblib
src/report.py            reports/<tag>/accuracy_report.md + accuracy.png
src/predict.py           forecast service used by the app
app/server.py            Flask API + app/static/index.html (Leaflet map)
src/era5_init.py         lat/lon/date -> cyclone centre + 24 h history + intensity from ARCO-ERA5
src/calibrate_era5.py    ERA5 -> IMD intensity calibration
tests/                   test_app.py (API), test_era5_flow.py (offline ERA5 chain)
```

## Run it (models are already trained)
```bash
pip install -r requirements.txt
python app/server.py            # open http://127.0.0.1:5000
python tests/test_app.py
```
App tabs:
* **Forecast (replay)** - pick any historical cyclone and the time the forecast is issued; the model sees only the
  past, the dashed line shows what really happened.
* **Lat / lon / date -> ERA5** - type a cyclone position and a UTC date/time. The server reads **Google ARCO-ERA5**,
  finds the pressure minimum near your point, tracks it back up to 24 h, takes wind / pressure / pressure drop
  from the fields, and runs the +6...+48 h forecast with start, end, land crossing and speed. Optional boxes let
  you give the known wind (kt) and central pressure (hPa) to rescale ERA5's weaker values.
* **Paste fixes** - give the last few fixes yourself (time, lat, lon, wind, pressure); no ERA5 needed.
* **Model accuracy** - tables of the measured accuracy.

The ERA5 tab needs internet access to Google Cloud Storage from the machine running the server
(`pip install xarray "zarr>=3" gcsfs dask`). ARCO lags real time by a few months, so the date must be in the past.
Use `ARCO_STORE=<url>` to point at another store. Offline test of this chain (synthetic cyclone, no network):
`python tests/test_era5_flow.py`.

## Re-train from the workbook
```bash
python src/clean_data.py --xlsx "Best Tracks (Data) 1982-2026.xlsx" --out data/tracks_clean.csv
python src/lifecycle.py
python src/train.py && python src/report.py track        # ~15 min on 2 CPU cores
```

## Use Google ARCO-ERA5 (wind, pressure, pressure drop, SST, shear, steering flow)
```bash
pip install xarray "zarr>=3" gcsfs dask
python src/fetch_arco_era5.py --max-hours 5        # quick test first
python src/fetch_arco_era5.py                      # full run; resumes from data/era5_cache
python src/train.py --use-era5 && python src/report.py era5
CYCLONE_MODEL=era5 python app/server.py
```
To isolate the environmental steering-wind components as the only added ERA5 features:
```bash
python src/train.py --steering-only --tag era5_wind
CYCLONE_MODEL=era5_wind python app/server.py
```
The steering-only training mode uses `era_steer_u_ms` and `era_steer_v_ms` from the full
`data/era5_features.csv` extraction; other ERA5 fields are excluded from X.
* Anonymous read of `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (change with `--store`).
* The store keeps one global field per hour, so the full 1982-2026 run reads on the order of **100 GB**. Use
  `--stride 6` (only 00/06/12/18 UTC fixes) and/or `--no-shear` to cut it, or run it on a Google Cloud VM.
* ERA5 has no "pressure drop" variable: `era_dp_hpa` = mean MSLP in the 5-8 deg ring minus the minimum MSLP within 2.5 deg.
* ERA5 (0.25 deg) under-represents cyclone winds; the model learns the correction against IMD's values.
* The extraction code was tested on a synthetic Zarr store with the same layout, **not on the real bucket**
  (it was not reachable from the build environment). If a variable name differs, the script prints the names present.
* **Calibration** (recommended once the fetch is done): `python src/calibrate_era5.py` fits ERA5 -> IMD maps for
  wind, pressure and pressure drop and writes `data/era5_calibration.json`; the lat/lon/date tab then applies it
  automatically. Without it the tab uses raw ERA5 values and says so (ERA5 under-represents intensity).
* For live use you need the same fields for *now*: ARCO lags real time, so use ECMWF open data
  (`ecmwf-opendata`, IFS step 0) and call `patch_features()` from `fetch_arco_era5.py` on it.

## Definitions
* **Start / end**: first / last best-track fix (genesis as depression, dissipation or merge). In forecasts, the
  predicted end is the first lead time where P(still a tracked system) < 50 %.
* **Land crossing**: the (interpolated, 15-minute) path is tested against a 1 km land mask; small islands
  (Andaman, Lakshadweep ...) are ignored via a 1x1 degree land-fraction filter, sea gaps shorter than 3 h are
  bridged. *Time to cross land* = landfall to exit to sea, or to dissipation when the system dies inland.
* **Speed**: forward (translational) speed in km/h from successive fixes; wind speed given in kt and km/h.
* **Intensity classes** (IMD, wind kt): D 17-27, DD 28-33, CS 34-47, SCS 48-63, VSCS 64-89, ESCS 90-119, SuCS >= 120.

## Accuracy (cyclones never seen in training) - see `reports/track/accuracy_report.md`
5-fold CV by cyclone, 1982-2026 (model vs simple baseline):

| Lead | Track error | Baseline | Wind MAE | Persistence | Pressure MAE | Persistence |
|---|---|---|---|---|---|---|
| +6 h | 51 km | 59 | 3.0 kt | 3.3 | 1.8 hPa | 2.0 |
| +24 h | 167 km | 219 | 8.7 kt | 11.6 | 5.3 hPa | 6.9 |
| +48 h | 338 km | 456 | 15.0 kt | 19.2 | 9.3 hPa | 11.7 |

Land crossing: 71 % of observed landfalls detected, 20 % false alarms, landfall time within +-6 h in 69 % of hits.

## Limits
* Roughly 420 cyclones is small data. 48 h errors are large, and this is **not** a replacement for IMD / JTWC
  advisories or NWP guidance. Use it for study and prototyping.
* Systems that are already over land, or whose history is shorter than 6 h, get weaker forecasts.
* Day/month swaps from Excel were corrected automatically in 204 date cells (`reports/data_quality.json`); the 2026
  sheet is marked tentative by IMD.
* Forecasts only exist where the system was still being tracked, so "dissipation" is learned by a separate
  still-active classifier rather than from the regression targets.
