from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from .models import load_compatible_state, make_model, parameter_count
from .train import forecast_test


NEURAL_MODELS = (
    "cnn",
    "lstm",
    "tcn",
    "ebamtn",
    "cpmtcn",
    "cpmtcn_no_exchange",
    "cpmtcn_soft",
)


def benchmark_efficiency(data, config: dict, output_root: str | Path) -> pd.DataFrame:
    """Benchmark end-to-end batched test inference from saved checkpoints."""
    output_root = Path(output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the configured efficiency benchmark")
    rows: list[dict] = []
    for model_name in NEURAL_MODELS:
        for seed in config["training"]["seeds"]:
            run_path = output_root / model_name / f"seed_{seed}" / "run.json"
            checkpoint_path = (
                Path(config["project"]["checkpoint_dir"]) / f"{model_name}_seed{seed}.pt"
            )
            run = json.loads(run_path.read_text(encoding="utf-8"))
            model = make_model(model_name, data, config).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            load_compatible_state(model, checkpoint["model"], model_name)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            forecasts = forecast_test(model, data, config)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            samples = len(forecasts["prediction"])
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "parameters_or_units": parameter_count(model),
                    "units_label": "trainable parameters",
                    "training_seconds": run.get("training_seconds"),
                    "inference_seconds": elapsed,
                    "samples": samples,
                    "milliseconds_per_sample": 1000 * elapsed / samples,
                    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                }
            )
            del forecasts, checkpoint, model
            torch.cuda.empty_cache()

    for model_name in ("ridge", "lightgbm"):
        path = output_root / model_name / "seed_42" / "efficiency.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model_name,
                    "seed": 42,
                    "parameters_or_units": record["model_units"],
                    "units_label": record["model_units_label"],
                    "training_seconds": record["training_seconds"],
                    "inference_seconds": record["inference_seconds"],
                    "samples": record["samples"],
                    "milliseconds_per_sample": record["milliseconds_per_sample"],
                    "peak_gpu_memory_mb": 0.0,
                }
            )

    chronos_path = output_root / "chronos2" / "seed_0" / "run.json"
    if chronos_path.exists():
        record = json.loads(chronos_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": "chronos2",
                "seed": 0,
                "parameters_or_units": 120_000_000,
                "units_label": "pretrained parameters",
                "training_seconds": 0.0,
                "inference_seconds": record["inference_seconds"],
                "samples": record["samples"],
                "milliseconds_per_sample": 1000
                * record["inference_seconds"]
                / record["samples"],
                "peak_gpu_memory_mb": pd.NA,
            }
        )

    for model_name in ("persistence", "seasonal_naive"):
        rows.append(
            {
                "model": model_name,
                "seed": pd.NA,
                "parameters_or_units": 0,
                "units_label": "parameters",
                "training_seconds": 0.0,
                "inference_seconds": pd.NA,
                "samples": 11_823,
                "milliseconds_per_sample": pd.NA,
                "peak_gpu_memory_mb": 0.0,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_root / "efficiency.csv", index=False)
    aggregate = (
        result.groupby(["model", "units_label"], as_index=False, dropna=False)
        .agg(
            runs=("model", "size"),
            parameters_or_units=("parameters_or_units", "mean"),
            training_seconds_mean=("training_seconds", "mean"),
            milliseconds_per_sample_mean=("milliseconds_per_sample", "mean"),
            peak_gpu_memory_mb_mean=("peak_gpu_memory_mb", "mean"),
        )
    )
    aggregate.to_csv(output_root / "efficiency_aggregate.csv", index=False)
    return result
