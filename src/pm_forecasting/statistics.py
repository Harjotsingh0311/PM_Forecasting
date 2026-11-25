from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def _load_forecast(output_root: Path, model: str, seed: int) -> dict[str, np.ndarray]:
    archive = np.load(output_root / model / f"seed_{seed}" / "forecasts.npz")
    return {name: archive[name] for name in archive.files}


def _reshape_by_origin(forecast: dict[str, np.ndarray], n_stations: int) -> tuple[np.ndarray, ...]:
    n_origins = len(forecast["origins"])
    expected = n_origins * n_stations
    if forecast["prediction"].shape[0] != expected:
        raise ValueError("Forecast ordering does not match origin-major/station-minor protocol")
    shape = (n_origins, n_stations) + forecast["prediction"].shape[1:]
    return (
        forecast["prediction"].reshape(shape),
        forecast["truth"].reshape(shape),
        forecast["mask"].reshape(shape),
    )


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int,
    origin_step: int = 3,
) -> tuple[float, float, float]:
    """Harvey-Leybourne-Newbold corrected DM test with Bartlett long-run variance."""
    difference = np.asarray(loss_a - loss_b, dtype=np.float64)
    difference = difference[np.isfinite(difference)]
    n = len(difference)
    if n < 10:
        return np.nan, np.nan, np.nan
    centered = difference - difference.mean()
    max_lag = int(math.ceil((horizon - 1) / origin_step))
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1 - lag / (max_lag + 1)
        long_run_variance += 2 * weight * covariance
    if long_run_variance <= 0:
        return np.nan, np.nan, float(difference.mean())
    statistic = difference.mean() / math.sqrt(long_run_variance / n)
    effective_horizon = max(1, int(math.ceil(horizon / origin_step)))
    correction = math.sqrt(
        max(n + 1 - 2 * effective_horizon + effective_horizon * (effective_horizon - 1) / n, 0)
        / n
    )
    statistic *= correction
    p_value = float(2 * student_t.sf(abs(statistic), df=n - 1))
    return float(statistic), p_value, float(difference.mean())


def run_dm_tests(
    data,
    output_root: str | Path,
    candidate: str = "cpmtcn",
    baseline: str = "tcn",
    seeds: tuple[int, ...] = (42, 123),
) -> pd.DataFrame:
    output_root = Path(output_root)
    rows = []
    for seed in seeds:
        candidate_forecast = _load_forecast(output_root, candidate, seed)
        baseline_forecast = _load_forecast(output_root, baseline, seed)
        candidate_prediction, truth, mask = _reshape_by_origin(
            candidate_forecast, len(data.stations)
        )
        baseline_prediction, baseline_truth, baseline_mask = _reshape_by_origin(
            baseline_forecast, len(data.stations)
        )
        if not np.allclose(truth, baseline_truth, equal_nan=True):
            raise ValueError("Candidate and baseline forecasts do not share targets")
        for pollutant, pollutant_name in enumerate(("PM2.5", "PM10")):
            for horizon in (6, 24):
                index = horizon - 1
                valid = mask[:, :, index, pollutant] & baseline_mask[:, :, index, pollutant]
                candidate_error = (candidate_prediction[:, :, index, pollutant] - truth[:, :, index, pollutant]) ** 2
                baseline_error = (baseline_prediction[:, :, index, pollutant] - truth[:, :, index, pollutant]) ** 2
                candidate_loss = np.nanmean(np.where(valid, candidate_error, np.nan), axis=1)
                baseline_loss = np.nanmean(np.where(valid, baseline_error, np.nan), axis=1)
                statistic, p_value, mean_difference = diebold_mariano(
                    candidate_loss, baseline_loss, horizon
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "seed": seed,
                        "pollutant": pollutant_name,
                        "horizon": horizon,
                        "origins": int(np.isfinite(candidate_loss - baseline_loss).sum()),
                        "mean_squared_error_difference": mean_difference,
                        "dm_hln_statistic": statistic,
                        "p_value_two_sided": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "diebold_mariano.csv", index=False)
    return result


def run_cpge_day_bootstrap(
    data,
    output_root: str | Path,
    seeds: tuple[int, ...] = (42, 123),
    repetitions: int = 2000,
) -> pd.DataFrame:
    output_root = Path(output_root)
    rng = np.random.default_rng(2026)
    rows = []
    for seed in seeds:
        gated = _load_forecast(output_root, "cpmtcn", seed)
        plain = _load_forecast(output_root, "cpmtcn_no_exchange", seed)
        gated_prediction, truth, mask = _reshape_by_origin(gated, len(data.stations))
        plain_prediction, _, plain_mask = _reshape_by_origin(plain, len(data.stations))
        dates = pd.DatetimeIndex(data.timestamps[gated["origins"]]).date
        unique_dates = np.unique(dates)
        day_indices = [np.flatnonzero(dates == day) for day in unique_dates]
        for pollutant, pollutant_name in enumerate(("PM2.5", "PM10")):
            for horizon in (1, 6, 12, 24):
                index = horizon - 1
                valid = mask[:, :, index, pollutant] & plain_mask[:, :, index, pollutant]
                gated_sq = np.where(
                    valid,
                    (gated_prediction[:, :, index, pollutant] - truth[:, :, index, pollutant]) ** 2,
                    np.nan,
                )
                plain_sq = np.where(
                    valid,
                    (plain_prediction[:, :, index, pollutant] - truth[:, :, index, pollutant]) ** 2,
                    np.nan,
                )
                observed = float(np.sqrt(np.nanmean(gated_sq)) - np.sqrt(np.nanmean(plain_sq)))
                samples = np.empty(repetitions, dtype=np.float64)
                for repetition in range(repetitions):
                    selected_days = rng.integers(0, len(day_indices), len(day_indices))
                    selected = np.concatenate([day_indices[day] for day in selected_days])
                    samples[repetition] = np.sqrt(np.nanmean(gated_sq[selected])) - np.sqrt(
                        np.nanmean(plain_sq[selected])
                    )
                rows.append(
                    {
                        "seed": seed,
                        "pollutant": pollutant_name,
                        "horizon": horizon,
                        "rmse_delta_gated_minus_no_exchange": observed,
                        "ci95_low": float(np.quantile(samples, 0.025)),
                        "ci95_high": float(np.quantile(samples, 0.975)),
                        "bootstrap_days": len(unique_dates),
                        "repetitions": repetitions,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "cpge_day_bootstrap.csv", index=False)
    return result


def run_physical_output_bootstrap(
    data,
    output_root: str | Path,
    seeds: tuple[int, ...] = (42, 123),
    repetitions: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare hard and soft PM10>=PM2.5 output strategies by whole-day resampling."""
    output_root = Path(output_root)
    rng = np.random.default_rng(2027)
    bootstrap_rows: list[dict] = []
    violation_rows: list[dict] = []
    for seed in seeds:
        soft = _load_forecast(output_root, "cpmtcn_soft", seed)
        hard = _load_forecast(output_root, "cpmtcn", seed)
        soft_prediction, truth, soft_mask = _reshape_by_origin(soft, len(data.stations))
        hard_prediction, hard_truth, hard_mask = _reshape_by_origin(hard, len(data.stations))
        if not np.allclose(truth, hard_truth, equal_nan=True):
            raise ValueError("Hard and soft physical-output runs do not share targets")
        dates = pd.DatetimeIndex(data.timestamps[soft["origins"]]).date
        unique_dates = np.unique(dates)
        day_indices = [np.flatnonzero(dates == day) for day in unique_dates]
        for horizon in (1, 6, 12, 24):
            index = horizon - 1
            pair_valid = soft_mask[:, :, index].all(axis=-1) & hard_mask[:, :, index].all(axis=-1)
            for model_name, prediction in (("hard", hard_prediction), ("soft", soft_prediction)):
                difference = prediction[:, :, index, 0] - prediction[:, :, index, 1]
                violations = pair_valid & (difference > 0)
                violation_rows.append(
                    {
                        "seed": seed,
                        "output_strategy": model_name,
                        "horizon": horizon,
                        "valid_pairs": int(pair_valid.sum()),
                        "violation_rate": float(violations.sum() / pair_valid.sum()),
                        "mean_violation_magnitude": float(
                            difference[violations].mean() if violations.any() else 0.0
                        ),
                    }
                )
            for pollutant, pollutant_name in enumerate(("PM2.5", "PM10")):
                valid = soft_mask[:, :, index, pollutant] & hard_mask[:, :, index, pollutant]
                soft_sq = np.where(
                    valid,
                    (soft_prediction[:, :, index, pollutant] - truth[:, :, index, pollutant]) ** 2,
                    np.nan,
                )
                hard_sq = np.where(
                    valid,
                    (hard_prediction[:, :, index, pollutant] - truth[:, :, index, pollutant]) ** 2,
                    np.nan,
                )
                observed = float(np.sqrt(np.nanmean(soft_sq)) - np.sqrt(np.nanmean(hard_sq)))
                samples = np.empty(repetitions, dtype=np.float64)
                for repetition in range(repetitions):
                    selected_days = rng.integers(0, len(day_indices), len(day_indices))
                    selected = np.concatenate([day_indices[day] for day in selected_days])
                    samples[repetition] = np.sqrt(np.nanmean(soft_sq[selected])) - np.sqrt(
                        np.nanmean(hard_sq[selected])
                    )
                bootstrap_rows.append(
                    {
                        "seed": seed,
                        "pollutant": pollutant_name,
                        "horizon": horizon,
                        "rmse_delta_soft_minus_hard": observed,
                        "ci95_low": float(np.quantile(samples, 0.025)),
                        "ci95_high": float(np.quantile(samples, 0.975)),
                        "bootstrap_days": len(unique_dates),
                        "repetitions": repetitions,
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    violations = pd.DataFrame(violation_rows)
    bootstrap.to_csv(output_root / "physical_output_bootstrap.csv", index=False)
    violations.to_csv(output_root / "physical_output_violations.csv", index=False)
    return bootstrap, violations

