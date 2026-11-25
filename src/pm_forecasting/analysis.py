from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_cross_pollutant_statistics(data, output_dir: str | Path) -> dict[str, Path]:
    """Step-A descriptive statistics without making causal claims."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.DatetimeIndex(data.timestamps)
    fine = data.targets_raw[..., 0]
    total = data.targets_raw[..., 1]
    coarse = np.maximum(total - fine, 0)
    ratio = np.divide(fine, total, out=np.full_like(fine, np.nan), where=total > 0)

    records = []
    for station_idx, station in enumerate(data.stations):
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "station": station,
                "ratio": ratio[:, station_idx],
            }
        )
        frame["month"] = frame["timestamp"].dt.month
        frame["hour"] = frame["timestamp"].dt.hour
        records.append(frame)
    ratio_frame = pd.concat(records, ignore_index=True)
    ratio_month = ratio_frame.groupby(["station", "month"], as_index=False)["ratio"].agg(["mean", "median", "count"])
    ratio_hour = ratio_frame.groupby(["station", "hour"], as_index=False)["ratio"].agg(["mean", "median", "count"])
    month_path, hour_path = output_dir / "ratio_by_month.csv", output_dir / "ratio_by_hour.csv"
    ratio_month.to_csv(month_path, index=False)
    ratio_hour.to_csv(hour_path, index=False)

    correlation_rows = []
    for station_idx, station in enumerate(data.stations):
        station_fine = pd.Series(fine[:, station_idx])
        station_total = pd.Series(total[:, station_idx])
        station_coarse = pd.Series(coarse[:, station_idx])
        for lag in range(-48, 49):
            correlation_rows.append(
                {
                    "station": station,
                    "lag_hours": lag,
                    "pair": "PM2.5-coarse",
                    "correlation": station_fine.corr(station_coarse.shift(lag)),
                }
            )
            correlation_rows.append(
                {
                    "station": station,
                    "lag_hours": lag,
                    "pair": "PM2.5-PM10",
                    "correlation": station_fine.corr(station_total.shift(lag)),
                }
            )
    correlation_path = output_dir / "lagged_cross_correlation.csv"
    pd.DataFrame(correlation_rows).to_csv(correlation_path, index=False)
    return {"ratio_month": month_path, "ratio_hour": hour_path, "correlation": correlation_path}

