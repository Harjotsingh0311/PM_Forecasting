from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import metric_rows


def _last_observed(values: np.ndarray, time: int, station: int) -> np.ndarray:
    result = np.empty(2, dtype=np.float32)
    for pollutant in range(2):
        history = values[max(0, time - 168) : time, station, pollutant]
        valid = history[np.isfinite(history)]
        result[pollutant] = valid[-1] if len(valid) else np.nan
    return result


def run_naive_baselines(data, config: dict, output_dir: str | Path) -> pd.DataFrame:
    horizon = config["data"]["horizon"]
    stride = config["data"]["eval_stride"]
    origins = data.origins.test[::stride]
    truth_rows, mask_rows, persistence_rows, seasonal_rows = [], [], [], []
    for origin in origins:
        for station in range(len(data.stations)):
            truth = data.targets_raw[origin : origin + horizon, station]
            mask = data.target_mask[origin : origin + horizon, station]
            last = _last_observed(data.targets_raw, origin, station)
            persistence = np.broadcast_to(last, (horizon, 2)).copy()
            seasonal = data.targets_raw[origin - 24 : origin, station].copy()
            seasonal_missing = ~np.isfinite(seasonal)
            seasonal[seasonal_missing] = persistence[seasonal_missing]
            truth_rows.append(truth)
            mask_rows.append(mask)
            persistence_rows.append(persistence)
            seasonal_rows.append(seasonal)
    truth, mask = np.asarray(truth_rows), np.asarray(mask_rows)
    predictions = {
        "persistence": np.asarray(persistence_rows),
        "seasonal_naive": np.asarray(seasonal_rows),
    }
    rows: list[dict] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, prediction in predictions.items():
        rows.extend(metric_rows(truth, prediction, mask, name))
        np.savez_compressed(
            output_dir / f"{name}_forecasts.npz",
            origins=origins,
            truth=truth,
            prediction=prediction,
            mask=mask,
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "naive_metrics.csv", index=False)
    return result

