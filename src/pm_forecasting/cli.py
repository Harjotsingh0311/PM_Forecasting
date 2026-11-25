from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit import run_audit
from .analysis import run_cross_pollutant_statistics
from .baselines import run_naive_baselines
from .classical import run_classical_baseline
from .chronos2 import run_chronos2
from .cross_value import bootstrap_cross_value, run_cross_value, summarize_cross_value
from .config import load_config
from .data import load_prepared, prepare_data, save_prepared
from .diagnostics import build_diagnostic_figures, build_diagnostic_tables
from .evaluate import metric_rows, save_metrics
from .models import make_model
from .train import forecast_test, train_model
from .reporting import build_summary
from .statistics import run_cpge_day_bootstrap, run_dm_tests, run_physical_output_bootstrap


def _data(config: dict):
    cache, metadata = config["data"]["cache_path"], config["data"]["metadata_path"]
    if not Path(cache).exists():
        prepared = prepare_data(config)
        save_prepared(prepared, cache, metadata)
        return prepared
    return load_prepared(cache, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="PM2.5/PM10 forecasting benchmark")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("prepare")
    subparsers.add_parser("baselines")
    subparsers.add_parser("analyze")
    subparsers.add_parser("report")
    subparsers.add_parser("significance")
    classical_parser = subparsers.add_parser("classical")
    classical_parser.add_argument("--model", choices=["ridge", "lightgbm"], required=True)
    classical_parser.add_argument("--seed", type=int, default=42)
    chronos_parser = subparsers.add_parser("chronos2")
    chronos_parser.add_argument("--model-id", default="amazon/chronos-2")
    chronos_parser.add_argument("--batch-size", type=int, default=16)
    chronos_parser.add_argument("--chunk-size", type=int, default=128)
    cross_parser = subparsers.add_parser("cross-value")
    cross_parser.add_argument("--model", choices=["cnn", "lstm", "tcn"], required=True)
    cross_parser.add_argument("--target", choices=["pm25", "coarse"], required=True)
    cross_parser.add_argument("--inputs", choices=["own", "both"], required=True)
    cross_parser.add_argument("--seed", type=int, choices=[42, 123], required=True)
    subparsers.add_parser("cross-value-summary")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", choices=["cnn", "lstm", "tcn", "ebamtn", "cpmtcn", "cpmtcn_no_exchange", "cpmtcn_soft"], default="tcn")
    train_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model", choices=["cnn", "lstm", "tcn", "ebamtn", "cpmtcn", "cpmtcn_no_exchange", "cpmtcn_soft"], required=True)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "audit":
        result = run_audit(
            config["data"]["raw_dir"],
            Path(config["project"]["output_dir"]) / "audit.json",
            config["data"]["max_station_missing_pct"],
        )
        print(json.dumps({key: value for key, value in result.items() if key != "stations"}, indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        prepared = prepare_data(config)
        save_prepared(prepared, config["data"]["cache_path"], config["data"]["metadata_path"])
        print(json.dumps({"shape": prepared.features.shape, "stations": len(prepared.stations), "origins": {name: len(getattr(prepared.origins, name)) for name in ("train_a", "val_w", "train_b", "val_r", "test")}}, indent=2))
    elif args.command == "baselines":
        print(run_naive_baselines(_data(config), config, Path(config["project"]["output_dir"]) / "baselines").to_string(index=False))
    elif args.command == "analyze":
        paths = run_cross_pollutant_statistics(
            _data(config), Path(config["project"]["output_dir"]) / "analysis"
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "report":
        data = _data(config)
        metrics, figure = build_summary(config["project"]["output_dir"])
        tables = build_diagnostic_tables(data, config["project"]["output_dir"])
        figures = build_diagnostic_figures(data, config["project"]["output_dir"])
        print(metrics.to_string(index=False))
        print(f"Figure: {figure}")
        print(f"Diagnostic tables: {tables}")
        print(f"Diagnostic figures: {figures}")
    elif args.command == "classical":
        data = _data(config)
        output_dir = (
            Path(config["project"]["output_dir"]) / args.model / f"seed_{args.seed}"
        )
        metrics = run_classical_baseline(args.model, data, config, output_dir, args.seed)
        (output_dir / "run.json").write_text(
            json.dumps({"model": args.model, "seed": args.seed}, indent=2), encoding="utf-8"
        )
        print(metrics.to_string(index=False))
    elif args.command == "significance":
        data = _data(config)
        dm = run_dm_tests(data, config["project"]["output_dir"])
        bootstrap = run_cpge_day_bootstrap(data, config["project"]["output_dir"])
        physical, violations = run_physical_output_bootstrap(
            data, config["project"]["output_dir"]
        )
        print("Diebold-Mariano tests")
        print(dm.to_string(index=False))
        print("\nCPGE paired day bootstrap")
        print(bootstrap.to_string(index=False))
        print("\nSoft versus hard physical-output paired day bootstrap")
        print(physical.to_string(index=False))
        print("\nPhysical-ordering violations")
        print(violations.to_string(index=False))
    elif args.command == "chronos2":
        metrics = run_chronos2(
            _data(config),
            config,
            Path(config["project"]["output_dir"]) / "chronos2" / "seed_0",
            model_id=args.model_id,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        print(metrics.to_string(index=False))
    elif args.command == "cross-value":
        metrics = run_cross_value(
            args.model,
            args.target,
            args.inputs,
            args.seed,
            _data(config),
            config,
            Path(config["project"]["output_dir"]) / "cross_value",
        )
        print(metrics.to_string(index=False))
    elif args.command == "cross-value-summary":
        output_root = Path(config["project"]["output_dir"]) / "cross_value"
        summary = summarize_cross_value(output_root)
        bootstrap = bootstrap_cross_value(_data(config), output_root)
        print(summary.to_string(index=False))
        print("\nPaired whole-day bootstrap")
        print(bootstrap.to_string(index=False))
    elif args.command == "train":
        data = _data(config)
        result = train_model(args.model, data, config, args.seed)
        forecasts = forecast_test(result.pop("model"), data, config)
        output_dir = Path(config["project"]["output_dir"]) / args.model / f"seed_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "forecasts.npz", **forecasts)
        rows = metric_rows(forecasts["truth"], forecasts["prediction"], forecasts["mask"], args.model)
        save_metrics(rows, str(output_dir / "metrics.csv"))
        (output_dir / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        data = _data(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = make_model(args.model, data, config).to(device)
        checkpoint_path = Path(config["project"]["checkpoint_dir"]) / f"{args.model}_seed{args.seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        forecasts = forecast_test(model, data, config)
        output_dir = Path(config["project"]["output_dir"]) / args.model / f"seed_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "forecasts.npz", **forecasts)
        rows = metric_rows(forecasts["truth"], forecasts["prediction"], forecasts["mask"], args.model)
        print(save_metrics(rows, str(output_dir / "metrics.csv")).to_string(index=False))


if __name__ == "__main__":
    main()

