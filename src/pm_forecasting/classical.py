from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .evaluate import metric_rows


REPORTED_HORIZONS = (1, 6, 12, 24)
LAGS = (1, 6, 12, 24, 48, 168)
ROLLING_WINDOWS = (24, 168)


def _origin_station_pairs(origins: np.ndarray, n_stations: int) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(origins, n_stations), np.tile(np.arange(n_stations), len(origins))


def build_classical_features(data, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lag, rolling, weather/calendar and station features for classical baselines."""
    origin_values, stations = _origin_station_pairs(origins, len(data.stations))
    pieces = []
    for lag in LAGS:
        pieces.append(data.features[origin_values - lag, stations, :2])

    # Chunk rolling calculations to avoid materialising every 168-hour window at once.
    rolling = np.empty((len(origin_values), len(ROLLING_WINDOWS) * 4), dtype=np.float32)
    chunk_size = 8192
    for start in range(0, len(origin_values), chunk_size):
        stop = min(start + chunk_size, len(origin_values))
        batch_origins = origin_values[start:stop]
        batch_stations = stations[start:stop]
        columns = []
        for window in ROLLING_WINDOWS:
            offsets = np.arange(-window, 0, dtype=np.int64)
            history = data.features[
                batch_origins[:, None] + offsets[None, :], batch_stations[:, None], :2
            ]
            columns.extend([history.mean(axis=1), history.std(axis=1)])
        rolling[start:stop] = np.concatenate(columns, axis=1)
    pieces.append(rolling)

    # The most recent observed weather/calendar state is available at forecast time.
    pieces.append(data.features[origin_values - 1, stations, 4:])
    station_one_hot = np.eye(len(data.stations), dtype=np.float32)[stations]
    pieces.append(station_one_hot)
    features = np.concatenate(pieces, axis=1).astype(np.float32)
    return features, origin_values, stations


def _make_estimator(name: str, seed: int):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise RuntimeError(
                "LightGBM is not installed. Install the 'boosting' project extra first."
            ) from error
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown classical model: {name}")


def run_classical_baseline(
    name: str,
    data,
    config: dict,
    output_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    train_origins = data.origins.train[:: config["data"]["train_stride"]]
    test_origins = data.origins.test[:: config["data"]["eval_stride"]]
    x_train, train_origin_values, train_stations = build_classical_features(data, train_origins)
    x_test, test_origin_values, test_stations = build_classical_features(data, test_origins)
    n_samples = len(test_origin_values)
    prediction = np.full((n_samples, config["data"]["horizon"], 2), np.nan, dtype=np.float32)
    truth = np.full_like(prediction, np.nan)
    mask = np.zeros_like(prediction, dtype=bool)

    model_metadata = []
    training_seconds = 0.0
    inference_seconds = 0.0
    model_units = 0
    for horizon in REPORTED_HORIZONS:
        target_time_train = train_origin_values + horizon - 1
        target_time_test = test_origin_values + horizon - 1
        horizon_index = horizon - 1
        for pollutant in range(2):
            valid_train = data.target_mask[target_time_train, train_stations, pollutant]
            estimator = _make_estimator(name, seed)
            started = time.perf_counter()
            estimator.fit(
                x_train[valid_train],
                data.targets_norm[target_time_train[valid_train], train_stations[valid_train], pollutant],
            )
            training_seconds += time.perf_counter() - started
            started = time.perf_counter()
            predicted_norm = estimator.predict(x_test)
            inference_seconds += time.perf_counter() - started
            if name == "ridge":
                model_units += int(estimator.coef_.size + np.asarray(estimator.intercept_).size)
            else:
                model_dump = estimator.booster_.dump_model()

                def count_leaves(node: dict) -> int:
                    if "leaf_index" in node:
                        return 1
                    return count_leaves(node["left_child"]) + count_leaves(node["right_child"])

                model_units += sum(
                    count_leaves(tree["tree_structure"]) for tree in model_dump["tree_info"]
                )
            means = data.target_log_mean[test_stations, pollutant]
            stds = data.target_log_std[test_stations, pollutant]
            prediction[:, horizon_index, pollutant] = np.maximum(
                np.expm1(predicted_norm * stds + means), 0
            )
            truth[:, horizon_index, pollutant] = data.targets_raw[
                target_time_test, test_stations, pollutant
            ]
            mask[:, horizon_index, pollutant] = data.target_mask[
                target_time_test, test_stations, pollutant
            ]
            model_metadata.append(
                {
                    "model": name,
                    "seed": seed,
                    "pollutant": pollutant,
                    "horizon": horizon,
                    "train_samples": int(valid_train.sum()),
                    "features": int(x_train.shape[1]),
                }
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "forecasts.npz",
        origins=test_origins,
        station_indices=test_stations,
        prediction=prediction,
        truth=truth,
        mask=mask,
    )
    metrics = pd.DataFrame(metric_rows(truth, prediction, mask, name))
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(model_metadata).to_csv(output_dir / "models.csv", index=False)
    (output_dir / "efficiency.json").write_text(
        json.dumps(
            {
                "model": name,
                "seed": seed,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "samples": n_samples,
                "milliseconds_per_sample": 1000 * inference_seconds / n_samples,
                "model_units": model_units,
                "model_units_label": "coefficients" if name == "ridge" else "tree leaves",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics

