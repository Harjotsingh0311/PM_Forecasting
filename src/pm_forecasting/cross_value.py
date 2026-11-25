from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import training_time_mask
from .train import forecast_test, train_model


def make_cross_value_data(data, config: dict, target: str, inputs: str):
    if target not in {"pm25", "coarse"} or inputs not in {"own", "both"}:
        raise ValueError("target must be pm25/coarse and inputs must be own/both")
    result = copy.copy(data)
    fine_features = data.features[..., [0, 2]]
    coarse_features = data.cp_features
    common_features = data.features[..., 4:17]
    if inputs == "both":
        result.features = np.concatenate(
            [fine_features, coarse_features, common_features], axis=-1
        ).astype(np.float32)
    elif target == "pm25":
        result.features = np.concatenate([fine_features, common_features], axis=-1).astype(
            np.float32
        )
    else:
        result.features = np.concatenate([coarse_features, common_features], axis=-1).astype(
            np.float32
        )

    coarse_raw = np.maximum(data.targets_raw[..., 1] - data.targets_raw[..., 0], 0)
    coarse_mask = data.target_mask[..., 0] & data.target_mask[..., 1]
    coarse_norm = (
        np.log1p(coarse_raw) - data.coarse_log_mean[None, :]
    ) / data.coarse_log_std[None, :]
    result.targets_raw = np.stack([data.targets_raw[..., 0], coarse_raw], axis=-1).astype(
        np.float32
    )
    result.targets_norm = np.stack([data.targets_norm[..., 0], coarse_norm], axis=-1).astype(
        np.float32
    )
    result.targets_norm = np.nan_to_num(result.targets_norm, nan=0.0)
    result.target_mask = np.stack([data.target_mask[..., 0], coarse_mask], axis=-1)
    selected_channel = 0 if target == "pm25" else 1
    result.target_mask[..., 1 - selected_channel] = False

    train_times = training_time_mask(pd.DatetimeIndex(data.timestamps), config["splits"])
    coarse_raw_std = np.nanstd(np.where(coarse_mask, coarse_raw, np.nan)[train_times], axis=0)
    coarse_q90 = np.nanquantile(
        np.where(coarse_mask, coarse_raw, np.nan)[train_times], 0.90, axis=0
    )
    result.target_log_mean = np.column_stack(
        [data.target_log_mean[:, 0], data.coarse_log_mean]
    ).astype(np.float32)
    result.target_log_std = np.column_stack(
        [data.target_log_std[:, 0], data.coarse_log_std]
    ).astype(np.float32)
    result.target_raw_std = np.column_stack(
        [data.target_raw_std[:, 0], coarse_raw_std]
    ).astype(np.float32)
    result.target_q90 = np.column_stack([data.target_q90[:, 0], coarse_q90]).astype(
        np.float32
    )
    # Generic models ignore cp_features; keep a correctly shaped placeholder for the store.
    result.cp_features = np.zeros((*result.features.shape[:2], 2), dtype=np.float32)
    return result, selected_channel


def _single_target_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    model: str,
    target: str,
    channel: int,
) -> pd.DataFrame:
    rows = []
    for horizon in (1, 6, 12, 24):
        index = horizon - 1
        valid = mask[:, index, channel]
        y, yhat = truth[valid, index, channel], prediction[valid, index, channel]
        error = yhat - y
        denominator = np.sum((y - y.mean()) ** 2)
        rows.append(
            {
                "model": model,
                "target": target,
                "horizon": horizon,
                "n": int(valid.sum()),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "r2": float(1 - np.sum(error**2) / denominator),
                "bias": float(np.mean(error)),
            }
        )
    return pd.DataFrame(rows)


def run_cross_value(
    model_name: str,
    target: str,
    inputs: str,
    seed: int,
    data,
    config: dict,
    output_root: str | Path,
) -> pd.DataFrame:
    experiment_data, channel = make_cross_value_data(data, config, target, inputs)
    experiment_config = copy.deepcopy(config)
    experiment_config["training"]["max_epochs"] = 15
    experiment_config["training"]["patience"] = 5
    experiment_config["data"]["train_stride"] = 4
    run_name = f"{model_name}_{target}_{inputs}"
    experiment_config["project"]["checkpoint_dir"] = str(
        Path("checkpoints") / "cross_value" / run_name
    )
    result = train_model(model_name, experiment_data, experiment_config, seed)
    forecasts = forecast_test(result.pop("model"), experiment_data, experiment_config)
    output_dir = Path(output_root) / run_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "forecasts.npz", **forecasts)
    metrics = _single_target_metrics(
        forecasts["truth"],
        forecasts["prediction"],
        forecasts["mask"],
        model_name,
        target,
        channel,
    )
    metrics["inputs"] = inputs
    metrics["seed"] = seed
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                **result,
                "model": model_name,
                "target": target,
                "inputs": inputs,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def summarize_cross_value(output_root: str | Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in Path(output_root).glob("*/seed_*/metrics.csv")]
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    pivot = metrics.pivot_table(
        index=["model", "target", "seed", "horizon"], columns="inputs", values="rmse"
    ).reset_index()
    if {"own", "both"}.issubset(pivot.columns):
        pivot["rmse_change_pct_with_other_fraction"] = 100 * (
            pivot["both"] / pivot["own"] - 1
        )
    path = Path(output_root) / "summary.csv"
    pivot.to_csv(path, index=False)
    return pivot


def bootstrap_cross_value(
    data,
    output_root: str | Path,
    repetitions: int = 2000,
) -> pd.DataFrame:
    output_root = Path(output_root)
    rng = np.random.default_rng(20261125)
    rows = []
    for model in ("cnn", "lstm", "tcn"):
        for target, channel in (("pm25", 0), ("coarse", 1)):
            for seed in (42, 123):
                own = np.load(
                    output_root / f"{model}_{target}_own" / f"seed_{seed}" / "forecasts.npz"
                )
                both = np.load(
                    output_root / f"{model}_{target}_both" / f"seed_{seed}" / "forecasts.npz"
                )
                n_origins = len(own["origins"])
                n_stations = len(data.stations)
                dates = pd.DatetimeIndex(data.timestamps[own["origins"]]).date
                unique_dates = np.unique(dates)
                day_indices = [np.flatnonzero(dates == day) for day in unique_dates]
                shape = (n_origins, n_stations, 24, 2)
                own_prediction = own["prediction"].reshape(shape)
                both_prediction = both["prediction"].reshape(shape)
                truth = own["truth"].reshape(shape)
                mask = own["mask"].reshape(shape)
                for horizon in (1, 6, 12, 24):
                    index = horizon - 1
                    valid = mask[:, :, index, channel]
                    own_sq = np.where(
                        valid,
                        (own_prediction[:, :, index, channel] - truth[:, :, index, channel]) ** 2,
                        np.nan,
                    )
                    both_sq = np.where(
                        valid,
                        (both_prediction[:, :, index, channel] - truth[:, :, index, channel]) ** 2,
                        np.nan,
                    )
                    observed = float(np.sqrt(np.nanmean(both_sq)) - np.sqrt(np.nanmean(own_sq)))
                    samples = np.empty(repetitions)
                    for repetition in range(repetitions):
                        chosen_days = rng.integers(0, len(day_indices), len(day_indices))
                        chosen = np.concatenate([day_indices[day] for day in chosen_days])
                        samples[repetition] = np.sqrt(np.nanmean(both_sq[chosen])) - np.sqrt(
                            np.nanmean(own_sq[chosen])
                        )
                    rows.append(
                        {
                            "model": model,
                            "target": target,
                            "seed": seed,
                            "horizon": horizon,
                            "rmse_delta_both_minus_own": observed,
                            "ci95_low": float(np.quantile(samples, 0.025)),
                            "ci95_high": float(np.quantile(samples, 0.975)),
                            "detectable": bool(
                                np.quantile(samples, 0.025) > 0
                                or np.quantile(samples, 0.975) < 0
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "bootstrap_summary.csv", index=False)
    return result

