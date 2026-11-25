from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = (1, 6, 12, 24)
POLLUTANTS = ("PM2.5", "PM10")


def _load(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {name: archive[name] for name in archive.files}


def _forecast_files(output_root: Path) -> list[tuple[str, int, Path]]:
    result = []
    for path in output_root.glob("*/seed_*/forecasts.npz"):
        model = path.parent.parent.name
        seed = int(path.parent.name.removeprefix("seed_"))
        result.append((model, seed, path))
    for model in ("persistence", "seasonal_naive"):
        path = output_root / "baselines" / f"{model}_forecasts.npz"
        if path.exists():
            result.append((model, -1, path))
    return sorted(result)


def build_diagnostic_tables(data, output_root: str | Path) -> dict[str, Path]:
    output_root = Path(output_root)
    peak_rows, monthly_rows, station_rows = [], [], []
    ratio_rows, gate_month_rows, gate_hour_rows = [], [], []
    exchange_month_rows, exchange_hour_rows = [], []
    files = _forecast_files(output_root)
    for model, seed, path in files:
        forecast = _load(path)
        prediction, truth, mask = forecast["prediction"], forecast["truth"], forecast["mask"]
        stations = (
            forecast["station_indices"].astype(int)
            if "station_indices" in forecast
            else np.tile(np.arange(len(data.stations)), len(forecast["origins"]))
        )
        sample_origins = np.repeat(forecast["origins"], len(data.stations))
        for pollutant, pollutant_name in enumerate(POLLUTANTS):
            for horizon in HORIZONS:
                index = horizon - 1
                valid = mask[:, index, pollutant] & np.isfinite(prediction[:, index, pollutant])
                peak = valid & (truth[:, index, pollutant] > data.target_q90[stations, pollutant])
                if peak.any():
                    error = prediction[peak, index, pollutant] - truth[peak, index, pollutant]
                    peak_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "pollutant": pollutant_name,
                            "horizon": horizon,
                            "n": int(peak.sum()),
                            "peak_mae": float(np.mean(np.abs(error))),
                            "peak_bias": float(np.mean(error)),
                        }
                    )
                timestamps = pd.DatetimeIndex(data.timestamps[sample_origins + index])
                for station_index, station_name in enumerate(data.stations):
                    selected = valid & (stations == station_index)
                    if not selected.any():
                        continue
                    error = (
                        prediction[selected, index, pollutant]
                        - truth[selected, index, pollutant]
                    )
                    station_truth = truth[selected, index, pollutant]
                    denominator = np.square(station_truth - station_truth.mean()).sum()
                    station_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "station": station_name,
                            "pollutant": pollutant_name,
                            "horizon": horizon,
                            "n": int(selected.sum()),
                            "mae": float(np.mean(np.abs(error))),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "r2": float(1 - np.square(error).sum() / denominator)
                            if denominator > 0
                            else np.nan,
                            "bias": float(np.mean(error)),
                        }
                    )
                for month in sorted(set(timestamps.month)):
                    selected = valid & (timestamps.month == month)
                    if not selected.any():
                        continue
                    error = prediction[selected, index, pollutant] - truth[selected, index, pollutant]
                    monthly_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "pollutant": pollutant_name,
                            "horizon": horizon,
                            "month": int(month),
                            "n": int(selected.sum()),
                            "mae": float(np.mean(np.abs(error))),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "bias": float(np.mean(error)),
                        }
                    )

        if model in {"tcn", "cpmtcn", "cpmtcn_no_exchange", "chronos2"}:
            index = 23
            timestamps = pd.DatetimeIndex(data.timestamps[sample_origins + index])
            valid_ratio = mask[:, index].all(axis=1) & (truth[:, index, 1] > 0)
            predicted_ratio = np.divide(
                prediction[:, index, 0],
                prediction[:, index, 1],
                out=np.full(len(prediction), np.nan, dtype=np.float32),
                where=prediction[:, index, 1] > 0,
            )
            observed_ratio = truth[:, index, 0] / truth[:, index, 1]
            for month in sorted(set(timestamps.month)):
                selected = valid_ratio & (timestamps.month == month)
                ratio_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "month": int(month),
                        "observed_ratio": float(np.nanmean(observed_ratio[selected])),
                        "predicted_ratio": float(np.nanmean(predicted_ratio[selected])),
                        "n": int(selected.sum()),
                    }
                )

        if model == "cpmtcn" and {"gate_names", "gate_values"}.issubset(forecast):
            timestamps = pd.DatetimeIndex(data.timestamps[sample_origins])
            for column, gate_name in enumerate(forecast["gate_names"].astype(str)):
                for month in sorted(set(timestamps.month)):
                    selected = timestamps.month == month
                    gate_month_rows.append(
                        {
                            "seed": seed,
                            "gate": gate_name,
                            "month": int(month),
                            "mean_gate": float(forecast["gate_values"][selected, column].mean()),
                        }
                    )
                for hour in sorted(set(timestamps.hour)):
                    selected = timestamps.hour == hour
                    gate_hour_rows.append(
                        {
                            "seed": seed,
                            "gate": gate_name,
                            "hour": hour,
                            "mean_gate": float(forecast["gate_values"][selected, column].mean()),
                        }
                    )

            if "exchange_share_values" in forecast:
                for column, gate_name in enumerate(forecast["gate_names"].astype(str)):
                    for month in sorted(set(timestamps.month)):
                        selected = timestamps.month == month
                        exchange_month_rows.append(
                            {
                                "seed": seed,
                                "gate": gate_name,
                                "month": int(month),
                                "mean_exchange_share": float(
                                    forecast["exchange_share_values"][selected, column].mean()
                                ),
                            }
                        )
                    for hour in sorted(set(timestamps.hour)):
                        selected = timestamps.hour == hour
                        exchange_hour_rows.append(
                            {
                                "seed": seed,
                                "gate": gate_name,
                                "hour": int(hour),
                                "mean_exchange_share": float(
                                    forecast["exchange_share_values"][selected, column].mean()
                                ),
                            }
                        )

    paths = {
        "peak": output_root / "peak_metrics.csv",
        "monthly": output_root / "monthly_metrics.csv",
        "station": output_root / "per_station_metrics.csv",
        "ratio": output_root / "ratio_forecast_by_month.csv",
        "gates_month": output_root / "gates_by_month.csv",
        "gates_hour": output_root / "gates_by_hour.csv",
        "exchange_month": output_root / "exchange_share_by_month.csv",
        "exchange_hour": output_root / "exchange_share_by_hour.csv",
    }
    pd.DataFrame(peak_rows).to_csv(paths["peak"], index=False)
    pd.DataFrame(monthly_rows).to_csv(paths["monthly"], index=False)
    pd.DataFrame(station_rows).to_csv(paths["station"], index=False)
    pd.DataFrame(ratio_rows).to_csv(paths["ratio"], index=False)
    pd.DataFrame(gate_month_rows).to_csv(paths["gates_month"], index=False)
    pd.DataFrame(gate_hour_rows).to_csv(paths["gates_hour"], index=False)
    pd.DataFrame(exchange_month_rows).to_csv(paths["exchange_month"], index=False)
    pd.DataFrame(exchange_hour_rows).to_csv(paths["exchange_hour"], index=False)
    return paths


def build_diagnostic_figures(data, output_root: str | Path) -> dict[str, Path]:
    output_root = Path(output_root)
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    gate_path = output_root / "gates_by_month.csv"
    exchange_path = output_root / "exchange_share_by_month.csv"
    if gate_path.exists() and gate_path.stat().st_size:
        gates = pd.read_csv(gate_path)
        averaged = gates.groupby(["gate", "month"], as_index=False)["mean_gate"].mean()
        has_exchange = exchange_path.exists() and exchange_path.stat().st_size
        fig, axes = plt.subplots(2 if has_exchange else 1, 1, figsize=(9, 7 if has_exchange else 4.5))
        axes = np.atleast_1d(axes)
        axis = axes[0]
        for gate, group in averaged.groupby("gate"):
            axis.plot(group["month"], group["mean_gate"], marker="o", label=gate)
        axis.set(title="CPGE gate values by test month", xlabel="Month", ylabel="Mean gate value")
        axis.set_xticks(sorted(averaged["month"].unique()))
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, frameon=False, fontsize=8)
        if has_exchange:
            exchange = pd.read_csv(exchange_path)
            exchange_average = exchange.groupby(["gate", "month"], as_index=False)[
                "mean_exchange_share"
            ].mean()
            share_axis = axes[1]
            for gate, group in exchange_average.groupby("gate"):
                share_axis.plot(
                    group["month"], group["mean_exchange_share"], marker="o", label=gate
                )
            share_axis.set(
                title="Transferred-feature share by test month",
                xlabel="Month",
                ylabel="Mean exchange share",
            )
            share_axis.set_xticks(sorted(exchange_average["month"].unique()))
            share_axis.grid(alpha=0.25)
            share_axis.legend(ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        paths["gates"] = figure_dir / "cpge_gates_by_month.png"
        fig.savefig(paths["gates"], dpi=180)
        plt.close(fig)

    gate_hour_path = output_root / "gates_by_hour.csv"
    exchange_hour_path = output_root / "exchange_share_by_hour.csv"
    if gate_hour_path.exists() and gate_hour_path.stat().st_size:
        gates = pd.read_csv(gate_hour_path)
        averaged = gates.groupby(["gate", "hour"], as_index=False)["mean_gate"].mean()
        has_exchange = exchange_hour_path.exists() and exchange_hour_path.stat().st_size
        fig, axes = plt.subplots(
            2 if has_exchange else 1, 1, figsize=(10, 7 if has_exchange else 4.5)
        )
        axes = np.atleast_1d(axes)
        for gate, group in averaged.groupby("gate"):
            axes[0].plot(group["hour"], group["mean_gate"], label=gate)
        axes[0].set(
            title="CPGE gate values by forecast-origin hour",
            xlabel="Recorded local hour",
            ylabel="Mean gate value",
        )
        axes[0].set_xticks(range(0, 24, 3))
        axes[0].grid(alpha=0.25)
        axes[0].legend(ncol=3, frameon=False, fontsize=8)
        if has_exchange:
            exchange = pd.read_csv(exchange_hour_path)
            exchange_average = exchange.groupby(["gate", "hour"], as_index=False)[
                "mean_exchange_share"
            ].mean()
            for gate, group in exchange_average.groupby("gate"):
                axes[1].plot(
                    group["hour"], group["mean_exchange_share"], label=gate
                )
            axes[1].set(
                title="Transferred-feature share by forecast-origin hour",
                xlabel="Recorded local hour",
                ylabel="Mean exchange share",
            )
            axes[1].set_xticks(range(0, 24, 3))
            axes[1].grid(alpha=0.25)
            axes[1].legend(ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        paths["gates_hour"] = figure_dir / "cpge_gates_by_hour.png"
        fig.savefig(paths["gates_hour"], dpi=180)
        plt.close(fig)

    ratio_path = output_root / "ratio_forecast_by_month.csv"
    if ratio_path.exists() and ratio_path.stat().st_size:
        ratios = pd.read_csv(ratio_path)
        averaged = ratios.groupby(["model", "month"], as_index=False).agg(
            observed_ratio=("observed_ratio", "mean"), predicted_ratio=("predicted_ratio", "mean")
        )
        fig, axis = plt.subplots(figsize=(9, 4.5))
        observed = averaged.groupby("month", as_index=False)["observed_ratio"].mean()
        axis.plot(observed["month"], observed["observed_ratio"], color="black", linewidth=2.5, marker="o", label="observed")
        for model, group in averaged.groupby("model"):
            axis.plot(group["month"], group["predicted_ratio"], marker="o", label=model)
        axis.set(title="PM2.5/PM10 ratio at 24-hour horizon", xlabel="Target month", ylabel="Mean ratio")
        axis.set_xticks(sorted(averaged["month"].unique()))
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        paths["ratio"] = figure_dir / "ratio_observed_vs_predicted.png"
        fig.savefig(paths["ratio"], dpi=180)
        plt.close(fig)

    episode_models = ["tcn", "cpmtcn", "chronos2"]
    loaded = {}
    for model in episode_models:
        candidates = sorted((output_root / model).glob("seed_*/forecasts.npz"))
        if candidates:
            loaded[model] = _load(candidates[0])
    if loaded:
        reference = next(iter(loaded.values()))
        target_origin = np.datetime64("2025-11-01T00:00")
        origin_index = int(np.argmin(np.abs(data.timestamps[reference["origins"]] - target_origin)))
        station = 0
        row = origin_index * len(data.stations) + station
        horizon_times = pd.DatetimeIndex(
            data.timestamps[reference["origins"][origin_index] + np.arange(24)]
        )
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for pollutant, axis in enumerate(axes):
            axis.plot(horizon_times, reference["truth"][row, :, pollutant], color="black", linewidth=2.3, label="observed")
            for model, forecast in loaded.items():
                axis.plot(horizon_times, forecast["prediction"][row, :, pollutant], label=model)
            axis.set_ylabel(f"{POLLUTANTS[pollutant]} (µg/m³)")
            axis.grid(alpha=0.25)
        axes[0].legend(ncol=4, frameon=False)
        axes[-1].set_xlabel("Target time")
        fig.suptitle(f"Heating-season 24-hour forecast episode - station {data.stations[station]}")
        fig.tight_layout()
        paths["episode"] = figure_dir / "heating_season_episode.png"
        fig.savefig(paths["episode"], dpi=180)
        plt.close(fig)
    return paths

