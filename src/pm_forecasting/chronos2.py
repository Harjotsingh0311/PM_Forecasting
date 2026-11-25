from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import metric_rows


def _forward_fill_short(values: np.ndarray, limit: int = 3) -> np.ndarray:
    result = values.copy()
    for column in range(result.shape[1]):
        gap = 0
        for row in range(result.shape[0]):
            if np.isfinite(result[row, column]):
                gap = 0
            else:
                gap += 1
                if gap <= limit and row > 0 and np.isfinite(result[row - 1, column]):
                    result[row, column] = result[row - 1, column]
    return result


def _make_input(data, origin: int, station: int, input_length: int, horizon: int) -> dict:
    history = slice(origin - input_length, origin)
    future = slice(origin, origin + horizon)
    target = _forward_fill_short(data.targets_raw[history, station]).T.astype(np.float32)
    covariates = data.features[history, station, 4:17]
    names = [
        "temperature",
        "pressure",
        "precipitation",
        "rain",
        "wind_speed",
        "wind_east",
        "wind_north",
        "wind_unknown",
        "cloud_level",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    ]
    past_covariates = {
        name: covariates[:, index].astype(np.float32) for index, name in enumerate(names)
    }
    future_calendar = data.features[future, station, 13:17]
    future_covariates = {
        name: future_calendar[:, index].astype(np.float32)
        for index, name in enumerate(names[-4:])
    }
    return {
        "target": target,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }


def run_chronos2(
    data,
    config: dict,
    output_dir: str | Path,
    model_id: str = "amazon/chronos-2",
    batch_size: int = 16,
    chunk_size: int = 128,
) -> pd.DataFrame:
    try:
        from chronos import Chronos2Pipeline
    except ImportError as error:
        raise RuntimeError(
            "Chronos is not installed. Install the 'foundation' project extra first."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Chronos-2 is configured for CUDA inference on this project")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "partial_forecasts.npz"
    origins = data.origins.test[:: config["data"]["eval_stride"]]
    origin_values = np.repeat(origins, len(data.stations))
    station_values = np.tile(np.arange(len(data.stations)), len(origins))
    n_samples = len(origin_values)
    horizon = config["data"]["horizon"]
    prediction = np.full((n_samples, horizon, 2), np.nan, dtype=np.float32)
    completed = 0
    if partial_path.exists():
        partial = np.load(partial_path)
        if len(partial["prediction"]) == n_samples:
            prediction[:] = partial["prediction"]
            completed = int(partial["completed"])

    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.float16,
        cache_dir=config["data"].get("model_cache_dir", "data/model_cache"),
    )
    started = time.perf_counter()
    input_length = config["data"]["input_length"]
    for start in range(completed, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        inputs = [
            _make_input(data, int(origin_values[index]), int(station_values[index]), input_length, horizon)
            for index in range(start, stop)
        ]
        _, point_forecasts = pipeline.predict_quantiles(
            inputs,
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=batch_size,
            context_length=input_length,
            cross_learning=False,
        )
        prediction[start:stop] = np.stack(
            [
                forecast.numpy().T
                if tuple(forecast.shape) == (2, horizon)
                else forecast.numpy()
                for forecast in point_forecasts
            ]
        )
        prediction[start:stop] = np.maximum(prediction[start:stop], 0)
        np.savez_compressed(partial_path, prediction=prediction, completed=np.asarray(stop))
        print(
            json.dumps(
                {
                    "completed": stop,
                    "total": n_samples,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    target_offsets = np.arange(horizon)
    target_times = origin_values[:, None] + target_offsets[None, :]
    truth = data.targets_raw[target_times, station_values[:, None]]
    mask = data.target_mask[target_times, station_values[:, None]]
    np.savez_compressed(
        output_dir / "forecasts.npz",
        origins=origins,
        station_indices=station_values,
        prediction=prediction,
        truth=truth,
        mask=mask,
    )
    metrics = pd.DataFrame(metric_rows(truth, prediction, mask, "chronos2"))
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "model": "chronos2",
                "model_id": model_id,
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "samples": n_samples,
                "inference_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)
    return metrics

