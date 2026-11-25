from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_station_frames


def _longest_missing_run(series: pd.Series) -> int:
    missing = series.isna().to_numpy()
    if not missing.any():
        return 0
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return int((ends - starts).max())


def _temperature_peak_hour(frame: pd.DataFrame, month: int) -> float | None:
    data = frame.loc[frame["datetime"].dt.month == month, ["datetime", "temperature"]].dropna()
    if data.empty:
        return None
    data = data.assign(day=data["datetime"].dt.date)
    peak_rows = data.loc[data.groupby("day")["temperature"].idxmax()]
    return float(peak_rows["datetime"].dt.hour.median())


def run_audit(
    raw_dir: str | Path,
    output_path: str | Path | None = None,
    station_missing_cutoff_pct: float = 15.0,
) -> dict:
    timestamps, stations, frames = load_station_frames(raw_dir)
    station_rows = []
    all_rows = []
    for station, frame in zip(stations, frames, strict=True):
        valid_pair = frame[["PM2.5", "PM10"]].dropna()
        row = {
            "station": station,
            "rows": len(frame),
            "pm25_missing_pct": round(float(frame["PM2.5"].isna().mean() * 100), 4),
            "pm10_missing_pct": round(float(frame["PM10"].isna().mean() * 100), 4),
            "wind_direction_missing_pct": round(
                float(frame["wind_direction"].isna().mean() * 100), 4
            ),
            "pm25_longest_gap_hours": _longest_missing_run(frame["PM2.5"]),
            "pm10_longest_gap_hours": _longest_missing_run(frame["PM10"]),
            "pm25_gt_pm10_pct": round(
                float((valid_pair["PM2.5"] > valid_pair["PM10"]).mean() * 100), 6
            ),
            "january_temperature_peak_hour": _temperature_peak_hour(frame, 1),
            "july_temperature_peak_hour": _temperature_peak_hour(frame, 7),
        }
        station_rows.append(row)
        tagged = frame.copy()
        tagged["station"] = station
        all_rows.append(tagged)
    combined = pd.concat(all_rows, ignore_index=True)
    pair = combined[["PM2.5", "PM10"]].dropna()
    january_peak = float(np.median([row["january_temperature_peak_hour"] for row in station_rows]))
    july_peak = float(np.median([row["july_temperature_peak_hour"] for row in station_rows]))
    numeric = ["PM2.5", "PM10", "temperature", "rain", "pressure", "precipitation", "wind_speed"]
    precipitation_correlations = (
        combined[["precipitation", "temperature", "rain", "pressure"]]
        .corr(numeric_only=True)["precipitation"]
        .drop("precipitation")
        .round(6)
        .to_dict()
    )
    audit = {
        "station_count": len(stations),
        "rows_per_station": len(timestamps),
        "total_rows": len(combined),
        "start": str(timestamps[0]),
        "end": str(timestamps[-1]),
        "shared_complete_hourly_grid": True,
        "pm25_missing_pct": round(float(combined["PM2.5"].isna().mean() * 100), 4),
        "pm10_missing_pct": round(float(combined["PM10"].isna().mean() * 100), 4),
        "wind_direction_missing_pct": round(
            float(combined["wind_direction"].isna().mean() * 100), 4
        ),
        "pm25_gt_pm10_count": int((pair["PM2.5"] > pair["PM10"]).sum()),
        "pm25_gt_pm10_pct": round(float((pair["PM2.5"] > pair["PM10"]).mean() * 100), 6),
        "recommended_decomposition": "hard" if (pair["PM2.5"] > pair["PM10"]).mean() < 0.05 else "soft",
        "january_temperature_peak_hour_median": january_peak,
        "july_temperature_peak_hour_median": july_peak,
        "timestamp_interpretation": (
            "local_civil_clock" if 0.5 <= abs(july_peak - january_peak) <= 1.5 else "fixed_offset_or_utc"
        ),
        "station_missing_cutoff_pct": station_missing_cutoff_pct,
        "stations_excluded": [
            row["station"]
            for row in station_rows
            if max(row["pm25_missing_pct"], row["pm10_missing_pct"])
            > station_missing_cutoff_pct
        ],
        "quantiles": combined[numeric].quantile([0.5, 0.9, 0.99]).round(3).to_dict(),
        "precipitation_diagnostic": {
            "correlations": precipitation_correlations,
            "interpretation": (
                "The 0-100 column behaves more like relative humidity or a provider "
                "probability score than measured precipitation: it is inversely related "
                "to temperature and increases with rain. The upstream semantic remains "
                "unverified, so the project keeps the source column name."
            ),
        },
        "stations": station_rows,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit

