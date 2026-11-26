#!/usr/bin/env bash
# End-to-end: clean data -> (optional ERA5) -> train -> report -> start app
set -e
XLSX="${1:-data/Best_Tracks_1982-2026.xlsx}"
python src/clean_data.py --xlsx "$XLSX" --out data/tracks_clean.csv
python src/lifecycle.py
python src/train.py
python src/report.py track
python app/server.py
