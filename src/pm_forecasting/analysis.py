from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from sklearn.linear_model import Ridge


def _residualize(values: np.ndarray, controls: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return ridge residuals on valid rows and NaN elsewhere."""
    residuals = np.full(len(values), np.nan, dtype=np.float64)
    if valid.sum() < controls.shape[1] + 10:
        return residuals
    model = Ridge(alpha=1.0)
    model.fit(controls[valid], values[valid])
    residuals[valid] = values[valid] - model.predict(controls[valid])
    return residuals


def _nested_f_test(
    response: np.ndarray,
    own: np.ndarray,
    other: np.ndarray,
    controls: np.ndarray | None,
    max_lag: int,
) -> tuple[float, float, int, float]:
    """Compare own-history and own+other-history linear predictive models."""
    rows = np.arange(max_lag, len(response))
    y = response[rows]
    own_lags = np.column_stack([own[rows - lag] for lag in range(1, max_lag + 1)])
    other_lags = np.column_stack([other[rows - lag] for lag in range(1, max_lag + 1)])
    restricted = np.column_stack([np.ones(len(rows)), own_lags])
    if controls is not None:
        restricted = np.column_stack([restricted, controls[rows - 1]])
    unrestricted = np.column_stack([restricted, other_lags])
    valid = np.isfinite(y) & np.isfinite(unrestricted).all(axis=1)
    y, restricted, unrestricted = y[valid], restricted[valid], unrestricted[valid]
    if len(y) <= unrestricted.shape[1] + 1:
        return np.nan, np.nan, len(y), np.nan
    beta_r = np.linalg.lstsq(restricted, y, rcond=None)[0]
    beta_u = np.linalg.lstsq(unrestricted, y, rcond=None)[0]
    rss_r = float(np.square(y - restricted @ beta_r).sum())
    rss_u = float(np.square(y - unrestricted @ beta_u).sum())
    added = unrestricted.shape[1] - restricted.shape[1]
    denominator_df = len(y) - unrestricted.shape[1]
    statistic = max(0.0, ((rss_r - rss_u) / added) / (rss_u / denominator_df))
    p_value = float(f_distribution.sf(statistic, added, denominator_df))
    partial_r2 = max(0.0, (rss_r - rss_u) / rss_r) if rss_r > 0 else np.nan
    return statistic, p_value, len(y), partial_r2


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

    conditioned_rows = []
    granger_rows = []
    for station_idx, station in enumerate(data.stations):
        station_fine = np.log1p(fine[:, station_idx].astype(np.float64))
        station_coarse = np.log1p(coarse[:, station_idx].astype(np.float64))
        weather = data.features[:, station_idx, 4:13].astype(np.float64)
        calendar = data.features[:, station_idx, 13:17].astype(np.float64)
        lagged_weather = np.vstack([np.full((1, weather.shape[1]), np.nan), weather[:-1]])
        # Remove calendar structure and the effect of weather known one hour
        # earlier, matching the proposal's past-weather conditioning rule.
        controls = np.column_stack([lagged_weather, calendar])
        valid = (
            np.isfinite(station_fine)
            & np.isfinite(station_coarse)
            & np.isfinite(controls).all(axis=1)
        )
        fine_residual = _residualize(station_fine, controls, valid)
        coarse_residual = _residualize(station_coarse, controls, valid)
        fine_series, coarse_series = pd.Series(fine_residual), pd.Series(coarse_residual)
        for lag in range(-48, 49):
            conditioned_rows.append(
                {
                    "station": station,
                    "lag_hours": lag,
                    "pair": "PM2.5-coarse",
                    "correlation": fine_series.corr(coarse_series.shift(lag)),
                    "conditioning": "past-weather+calendar ridge residuals",
                }
            )

        # First-differenced log concentrations reduce trend/seasonality. The
        # nested F-test is Granger-style predictive evidence, never causality.
        fine_diff = np.diff(station_fine, prepend=np.nan)
        coarse_diff = np.diff(station_coarse, prepend=np.nan)
        for conditioned in (False, True):
            # _nested_f_test selects controls at t-1, so use the unshifted
            # weather/calendar matrix here.
            test_controls = np.column_stack([weather, calendar]) if conditioned else None
            for response, own, other, direction in (
                (coarse_diff, coarse_diff, fine_diff, "PM2.5 -> coarse"),
                (fine_diff, fine_diff, coarse_diff, "coarse -> PM2.5"),
            ):
                statistic, p_value, observations, partial_r2 = _nested_f_test(
                    response, own, other, test_controls, max_lag=24
                )
                granger_rows.append(
                    {
                        "station": station,
                        "direction": direction,
                        "conditioned_on_weather_calendar": conditioned,
                        "max_lag_hours": 24,
                        "observations": observations,
                        "f_statistic": statistic,
                        "p_value": p_value,
                        "partial_r2_added_history": partial_r2,
                    }
                )

    conditioned_path = output_dir / "weather_conditioned_cross_correlation.csv"
    granger_path = output_dir / "granger_predictive_tests.csv"
    pd.DataFrame(conditioned_rows).to_csv(conditioned_path, index=False)
    pd.DataFrame(granger_rows).to_csv(granger_path, index=False)
    return {
        "ratio_month": month_path,
        "ratio_hour": hour_path,
        "correlation": correlation_path,
        "conditioned_correlation": conditioned_path,
        "granger_predictive": granger_path,
    }

