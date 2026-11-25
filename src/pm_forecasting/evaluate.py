from __future__ import annotations

import numpy as np
import pandas as pd


def metric_rows(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    horizons: tuple[int, ...] = (1, 6, 12, 24),
) -> list[dict]:
    rows: list[dict] = []
    pollutant_names = ["PM2.5", "PM10"]
    for pollutant, pollutant_name in enumerate(pollutant_names):
        for horizon in horizons:
            index = horizon - 1
            valid = mask[:, index, pollutant] & np.isfinite(truth[:, index, pollutant])
            y, yhat = truth[valid, index, pollutant], prediction[valid, index, pollutant]
            error = yhat - y
            denominator = np.sum((y - y.mean()) ** 2)
            rows.append(
                {
                    "model": model_name,
                    "pollutant": pollutant_name,
                    "horizon": horizon,
                    "n": int(valid.sum()),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
                    "bias": float(np.mean(error)),
                }
            )
    return rows


def save_metrics(rows: list[dict], path: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path_obj = __import__("pathlib").Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path_obj, index=False)
    return frame

