from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit import run_audit
from .analysis import run_cross_pollutant_statistics
from .baselines import run_naive_baselines
from .config import load_config
from .data import load_prepared, prepare_data, save_prepared
from .evaluate import metric_rows, save_metrics
from .models import make_model
from .train import forecast_test, train_model
from .reporting import build_summary


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
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", choices=["cnn", "lstm", "tcn", "ebamtn", "cpmtcn", "cpmtcn_no_exchange"], default="tcn")
    train_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model", choices=["cnn", "lstm", "tcn", "ebamtn", "cpmtcn", "cpmtcn_no_exchange"], required=True)
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
        metrics, figure = build_summary(config["project"]["output_dir"])
        print(metrics.to_string(index=False))
        print(f"Figure: {figure}")
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

