from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_summary(output_root: str | Path) -> tuple[pd.DataFrame, Path]:
    output_root = Path(output_root)
    metric_files = list(output_root.glob("*/seed_*/metrics.csv"))
    baseline_file = output_root / "baselines" / "naive_metrics.csv"
    frames = []
    for path in metric_files:
        frame = pd.read_csv(path)
        frame["seed"] = int(path.parent.name.removeprefix("seed_"))
        frames.append(frame)
    if baseline_file.exists():
        baseline = pd.read_csv(baseline_file)
        baseline["seed"] = pd.NA
        frames.append(baseline)
    if not frames:
        raise FileNotFoundError("No metric CSV files found; run baselines or train a model first")
    metrics = pd.concat(frames, ignore_index=True)
    persistence = (
        metrics.loc[metrics["model"] == "persistence", ["pollutant", "horizon", "rmse"]]
        .drop_duplicates(["pollutant", "horizon"])
        .rename(columns={"rmse": "persistence_rmse"})
    )
    metrics = metrics.merge(persistence, on=["pollutant", "horizon"], how="left")
    metrics["skill_vs_persistence"] = 1 - metrics["rmse"] / metrics["persistence_rmse"]
    summary_path = output_root / "summary_metrics.csv"
    metrics.to_csv(summary_path, index=False)
    aggregate = (
        metrics.groupby(["model", "pollutant", "horizon"], as_index=False, dropna=False)
        .agg(
            seeds=("seed", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            skill_mean=("skill_vs_persistence", "mean"),
        )
    )
    aggregate.to_csv(output_root / "aggregate_metrics.csv", index=False)

    cpge = metrics[metrics["model"].isin(["cpmtcn", "cpmtcn_no_exchange"])].pivot_table(
        index=["seed", "pollutant", "horizon"], columns="model", values="rmse"
    )
    if {"cpmtcn", "cpmtcn_no_exchange"}.issubset(cpge.columns):
        cpge["rmse_delta_gated_minus_no_exchange"] = cpge["cpmtcn"] - cpge["cpmtcn_no_exchange"]
        cpge.reset_index().to_csv(output_root / "cpge_ablation.csv", index=False)

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "rmse_by_horizon.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for axis, pollutant in zip(axes, ["PM2.5", "PM10"], strict=True):
        subset = metrics[metrics["pollutant"] == pollutant]
        for model, group in subset.groupby("model"):
            group = group.groupby("horizon", as_index=False)["rmse"].mean().sort_values("horizon")
            axis.plot(group["horizon"], group["rmse"], marker="o", label=model)
        axis.set_title(pollutant)
        axis.set_xlabel("Forecast horizon (hours)")
        axis.set_ylabel("RMSE (µg/m³)")
        axis.grid(alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return metrics, figure_path

