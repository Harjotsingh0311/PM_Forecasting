from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.select_dtypes(include=["float", "floating"]).columns:
        values[column] = values[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    values = values.fillna("")
    headers = [str(column) for column in values.columns]
    rows = [[str(value) for value in row] for row in values.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


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

    run_overall = (
        metrics.groupby(["model", "pollutant", "seed"], as_index=False, dropna=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            bias=("bias", "mean"),
            skill_vs_persistence=("skill_vs_persistence", "mean"),
        )
    )
    overall = (
        run_overall.groupby(["model", "pollutant"], as_index=False, dropna=False)
        .agg(
            runs=("rmse", "size"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            bias_mean=("bias", "mean"),
            skill_mean=("skill_vs_persistence", "mean"),
        )
    )
    overall.to_csv(output_root / "overall_accuracy.csv", index=False)

    cpge = metrics[metrics["model"].isin(["cpmtcn", "cpmtcn_no_exchange"])].pivot_table(
        index=["seed", "pollutant", "horizon"], columns="model", values="rmse"
    )
    if {"cpmtcn", "cpmtcn_no_exchange"}.issubset(cpge.columns):
        cpge["rmse_delta_gated_minus_no_exchange"] = cpge["cpmtcn"] - cpge["cpmtcn_no_exchange"]
        cpge.reset_index().to_csv(output_root / "cpge_ablation.csv", index=False)

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "rmse_by_horizon.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for axis, pollutant in zip(axes, ["PM2.5", "PM10"], strict=True):
        subset = metrics[metrics["pollutant"] == pollutant]
        for model, group in subset.groupby("model"):
            group = group.groupby("horizon", as_index=False)["rmse"].mean().sort_values("horizon")
            axis.plot(group["horizon"], group["rmse"], marker="o", label=model)
        axis.set_title(pollutant)
        axis.set_xlabel("Forecast horizon (hours)")
        axis.set_ylabel("RMSE (µg/m³)")
        axis.grid(alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return metrics, figure_path


def build_project_report(output_root: str | Path, report_path: str | Path) -> Path:
    """Create the proposal's five-table report from finalized cached outputs."""
    output_root, report_path = Path(output_root), Path(report_path)
    overall = pd.read_csv(output_root / "overall_accuracy.csv")
    horizon = pd.read_csv(output_root / "aggregate_metrics.csv")
    peaks = pd.read_csv(output_root / "peak_metrics.csv")
    cross = pd.read_csv(output_root / "cross_value" / "summary.csv")
    cpge = pd.read_csv(output_root / "cpge_day_bootstrap.csv")
    physical = pd.read_csv(output_root / "physical_output_bootstrap.csv")
    efficiency = pd.read_csv(output_root / "efficiency_aggregate.csv")

    table1 = overall[
        ["model", "pollutant", "runs", "mae_mean", "rmse_mean", "r2_mean", "bias_mean", "skill_mean"]
    ].sort_values(["pollutant", "rmse_mean"])

    table2 = horizon.pivot_table(
        index="model", columns=["pollutant", "horizon"], values="rmse_mean"
    )
    table2.columns = [f"{pollutant} {int(hours)}h" for pollutant, hours in table2.columns]
    table2 = table2.reset_index()

    table3 = (
        peaks.groupby(["model", "pollutant"], as_index=False)
        .agg(
            horizons=("horizon", "nunique"),
            peak_mae_mean=("peak_mae", "mean"),
            peak_bias_mean=("peak_bias", "mean"),
        )
        .sort_values(["pollutant", "peak_mae_mean"])
    )

    table4 = (
        cross.groupby(["model", "target", "horizon"], as_index=False)
        .agg(
            own_rmse=("own", "mean"),
            both_rmse=("both", "mean"),
            change_pct=("rmse_change_pct_with_other_fraction", "mean"),
        )
        .sort_values(["target", "model", "horizon"])
    )

    cpge_table = cpge.rename(
        columns={
            "rmse_delta_gated_minus_no_exchange": "rmse_delta",
            "ci95_low": "ci_low",
            "ci95_high": "ci_high",
        }
    )[["seed", "pollutant", "horizon", "rmse_delta", "ci_low", "ci_high"]]
    cpge_table.insert(0, "comparison", "full - no exchange")
    physical_table = physical.rename(
        columns={
            "rmse_delta_soft_minus_hard": "rmse_delta",
            "ci95_low": "ci_low",
            "ci95_high": "ci_high",
        }
    )[["seed", "pollutant", "horizon", "rmse_delta", "ci_low", "ci_high"]]
    physical_table.insert(0, "comparison", "soft - hard")
    table5 = pd.concat([cpge_table, physical_table], ignore_index=True)

    report = f"""# Technical report: PM2.5 and PM10 forecasting

This report contains the five tables and four figures predeclared in the project proposal. All values come from cached test forecasts on the shared October-December 2025 origins. Neural results use seeds 42 and 123; fixed baselines retain their single configured run.

## Table 1. Overall accuracy

Metrics are arithmetic means across the reported 1, 6, 12 and 24-hour horizons. `skill_mean` is relative to persistence.

{_markdown_table(table1)}

## Table 2. RMSE by forecast horizon

Values are seed means in µg/m³.

{_markdown_table(table2)}

## Table 3. Peak-event error and bias

Targets above each station and pollutant's training 90th percentile define peak events. Values are averaged across horizons and seeds; the complete horizon-level table is `outputs/peak_metrics.csv`.

{_markdown_table(table3)}

## Table 4. Cross-pollutant value

Negative `change_pct` means that adding the other fraction reduced RMSE. Statistical interpretation must use `outputs/cross_value/bootstrap_summary.csv`, which resamples whole test days.

{_markdown_table(table4)}

## Table 5. CP-MTCN variants

RMSE deltas are paired whole-day-bootstrap comparisons in µg/m³. Negative values favor the first named strategy.

{_markdown_table(table5)}

## Efficiency appendix

Neural timings are end-to-end batched inference on the RTX 3050 and include decoding and transfer of predictions back to the CPU. Classical timings cover all eight pollutant/horizon estimators. A station-origin is one 24-hour, two-pollutant forecast request.

{_markdown_table(efficiency)}

Full station-level and month-level accuracy are available in `outputs/per_station_metrics.csv` and `outputs/monthly_metrics.csv`.

## Figure 1. Heating-season forecast episode

![Heating-season forecast episode](figures/heating_season_episode.png)

## Figure 2. Error versus forecast horizon

![RMSE by horizon](figures/rmse_by_horizon.png)

## Figure 3. CPGE gates and exchange share by month

![CPGE diagnostics](figures/cpge_gates_by_month.png)

## Figure 4. PM2.5/PM10 ratio by month

![Ratio diagnostic](figures/ratio_observed_vs_predicted.png)

The additional hour-of-day gate and exchange-share diagnostic is available as `figures/cpge_gates_by_hour.png`.

## Main findings

- LightGBM gives the lowest one-hour PM2.5 RMSE; zero-shot Chronos-2 gives the lowest one-hour PM10 RMSE.
- TCN is the strongest general trained model and leads PM10 at 6, 12 and 24 hours.
- CPGE improves both pollutants at one hour in both seeds, but the longer-horizon effect is not stable.
- The hard output has zero PM2.5-above-PM10 violations. The soft output trades occasional violations for some medium- and long-horizon gains.
- The controlled value experiment finds one cross-seed robust improvement: adding coarse history to the CNN PM2.5 model at one hour.

## Scope and limitations

The test block covers one autumn-to-winter transition. The project uses past weather rather than operational future weather forecasts, two neural seeds, point predictions, and station embeddings without a geographic graph. Granger-style tests describe incremental predictive information and do not establish causality.

## References

1. Vrbancic, G. *Air quality dataset*, Mendeley Data v2. DOI: 10.17632/p86hnykz3d.2.
2. Xie, Z.-A. et al. *Multi-pollutant air quality forecasting using bidirectional attention and multi-scale temporal networks*. Frontiers in Environmental Science 13:1623630, 2025.
3. Lu, J. and Sun, S. *A causality-aware spatiotemporal model for multi-region and multi-pollutant air quality forecasting*. arXiv:2509.21260, 2025.
4. Badouch, A. and Belhoucine, K. *Benchmarking Deep Learning for PM2.5 Forecasting in IoT-Based Smart Cities*. IJACSA 17(5), 2026.
5. Ansari, A. F. et al. *Chronos-2: From Univariate to Universal Forecasting*. arXiv:2510.15821, 2025.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path

