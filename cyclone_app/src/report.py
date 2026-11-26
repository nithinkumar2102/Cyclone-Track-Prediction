"""Write reports/<tag>/accuracy_report.md and accuracy.png from the saved metrics."""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def md_table(df, cols, names, fmt):
    out = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(names)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f.format(r[c]) if pd.notna(r[c]) else "–" for c, f in zip(cols, fmt)) + " |")
    return "\n".join(out)


def main(tag="track"):
    rep = ROOT / "reports" / tag
    cv = pd.read_csv(rep / "metrics_cv5.csv")
    ho = pd.read_csv(rep / "metrics_holdout_2023_2026.csv")
    sm = json.loads((rep / "summary.json").read_text())
    cols = ["horizon_h", "track_err_km", "base_track_err_km", "track_within_100km_pct", "wind_mae_kt", "base_wind_mae_kt",
            "wind_within_10kt_pct", "category_exact_pct", "pressure_mae_hpa", "base_pressure_mae_hpa",
            "pressure_within_5hpa_pct", "pdrop_mae_hpa", "active_acc_pct"]
    names = ["Lead (h)", "Track err (km)", "Baseline (km)", "Within 100 km", "Wind MAE (kt)", "Persist. (kt)",
             "Within ±10 kt", "Class correct", "Pressure MAE (hPa)", "Persist. (hPa)", "Within ±5 hPa",
             "ΔP MAE (hPa)", "Still-active acc."]
    fm = ["+{:.0f}", "{:.0f}", "{:.0f}", "{:.0f}%", "{:.1f}", "{:.1f}", "{:.0f}%", "{:.0f}%", "{:.1f}", "{:.1f}",
          "{:.0f}%", "{:.1f}", "{:.0f}%"]
    lf = lambda d: (f"| Model | {d['model']['cases']} | {d['model']['detection_rate_pct']:.0f}% | "
                    f"{d['model']['false_alarm_ratio_pct']:.0f}% | {d['model']['landfall_time_mae_h']:.1f} h | "
                    f"{d['model']['landfall_time_within_6h_pct']:.0f}% | {d['model']['landfall_position_mae_km']:.0f} km |\n"
                    f"| Baseline | {d['baseline']['cases']} | {d['baseline']['detection_rate_pct']:.0f}% | "
                    f"{d['baseline']['false_alarm_ratio_pct']:.0f}% | {d['baseline']['landfall_time_mae_h']:.1f} h | "
                    f"{d['baseline']['landfall_time_within_6h_pct']:.0f}% | {d['baseline']['landfall_position_mae_km']:.0f} km |")
    lfh = "| | Forecasts | Landfalls detected | False alarms | Landfall-time error | Within ±6 h | Position error |\n|---|---|---|---|---|---|---|\n"
    txt = f"""# Accuracy report ({tag} model)

Data: {sm['n_cyclones']} cyclones / {sm['n_fixes']} best-track fixes (IMD, 1982-2026). Every number below is measured on
cyclones the model was **not trained on**. Baselines: track = constant motion of the last 6 h; wind / pressure = persistence.

## 5-fold cross-validation (whole cyclones held out, all years)
{md_table(cv, cols, names, fm)}

## Forward hold-out: trained 1982-2022, tested on 2023-2026 ({sm['holdout_cyclones']} cyclones)
{md_table(ho, cols, names, fm)}

## Land-crossing accuracy (cross-validation)
{lfh}{lf(sm['landfall_cv'])}

## Land-crossing accuracy (hold-out 2023-2026)
{lfh}{lf(sm['landfall_holdout'])}

*Landfall-time / position error is for forecasts where both the forecast and the observed track cross the coast.*
"""
    (rep / "accuracy_report.md").write_text(txt)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, (m, b, t, u) in zip(ax, [("track_err_km", "base_track_err_km", "Track error", "km"),
                                    ("wind_mae_kt", "base_wind_mae_kt", "Wind MAE", "kt"),
                                    ("pressure_mae_hpa", "base_pressure_mae_hpa", "Central pressure MAE", "hPa"),
                                    ("active_acc_pct", "active_base_acc_pct", "'Still active' accuracy", "%")]):
        a.plot(cv.horizon_h, cv[m], "o-", label="model (CV)")
        a.plot(cv.horizon_h, cv[b], "s--", color="gray", label="baseline")
        a.plot(ho.horizon_h, ho[m], "^-", alpha=.7, label="model (2023-26)")
        a.set_title(t); a.set_xlabel("lead time (h)"); a.set_ylabel(u); a.grid(alpha=.3)
    ax[0].legend(); plt.tight_layout(); plt.savefig(rep / "accuracy.png", dpi=130)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "track")
