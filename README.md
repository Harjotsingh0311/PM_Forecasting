# PM2.5 and PM10 Forecasting

This repository implements the proposed 21-station, leak-free forecasting benchmark. Each model receives 168 hours of history and predicts PM2.5 and PM10 for the next 24 hours. The code is tuned for the local NVIDIA RTX 3050 6 GB laptop GPU.

## What is implemented

- Validation of the common 14,100-hour grid across all 21 station files.
- Causal stuck-sensor and spike flags, three-hour past-only forward filling, masked targets, training-only scaling, local calendar features, wind vector conversion, and cloud encoding.
- Exact chronological Train A, winter validation, Train B, recent validation, and test blocks.
- Automated causality and split-boundary tests.
- Persistence and 24-hour seasonal-naive baselines on common three-hour test origins.
- 1D-CNN, two-layer LSTM, vanilla TCN, lean EBAMTN re-implementation, CP-MTCN-lite, and the no-exchange CP-MTCN variant.
- Mixed-precision CUDA training, composite peak-aware loss, early stopping on both validation blocks, checkpoints, cached forecasts, and MAE/RMSE/R2/bias metrics.

## Environment

The checked machine has Python 3.13, PyTorch 2.11.0+cu128, CUDA available, and an RTX 3050 6 GB. The local `.venv` reuses those verified system packages to avoid downloading a second multi-gigabyte CUDA build.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

For a completely isolated installation on another NVIDIA machine:

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

Verify the GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

VS Code is configured to select `.venv` automatically. Open this folder, then use **Terminal > Run Task** for audit, preparation, tests, or baselines.

## Reproducible workflow

```powershell
# 1. Dataset checks and design decision report
.\.venv\Scripts\python.exe -m pm_forecasting.cli audit

# 2. Build the leak-free feature cache and saved origin indices
.\.venv\Scripts\python.exe -m pm_forecasting.cli prepare

# 3. Run leakage, boundary, and architecture tests
.\.venv\Scripts\python.exe -m pytest -q

# 4. Run non-neural floors
.\.venv\Scripts\python.exe -m pm_forecasting.cli baselines

# 4b. Descriptive cross-pollutant statistics
.\.venv\Scripts\python.exe -m pm_forecasting.cli analyze

# 5. Train one model/seed
.\.venv\Scripts\python.exe -m pm_forecasting.cli train --model tcn --seed 42
.\.venv\Scripts\python.exe -m pm_forecasting.cli train --model cpmtcn --seed 42

# Rebuild forecasts/metrics from an existing best checkpoint (no retraining)
.\.venv\Scripts\python.exe -m pm_forecasting.cli evaluate --model cpmtcn --seed 42

# 6. Merge completed metrics and draw the horizon plot
.\.venv\Scripts\python.exe -m pm_forecasting.cli report
```

Valid model names are `cnn`, `lstm`, `tcn`, `ebamtn`, `cpmtcn`, and `cpmtcn_no_exchange`. Repeat required runs with seeds 42 and 123.

To run or resume the complete core matrix unattended, use `powershell -ExecutionPolicy Bypass -File scripts/run_core.ps1`. Runs with an existing `run.json` are skipped. Do not launch two copies of the matrix simultaneously.

## Confirmed data facts

- 21 stations x 14,100 hourly rows = 296,100 records.
- Common grid: 2024-05-02 20:00 through 2025-12-11 07:00, without missing or duplicate timestamps.
- PM2.5 missing: about 1.65%; PM10 missing: about 1.66%; wind direction missing: about 49.17%.
- Only 3 jointly observed records have PM2.5 above PM10 (about 0.001%), so the hard physical output `PM10 = PM2.5 + nonnegative coarse part` is used.
- E803 has the largest missing interval (1,001 hours in Train B) and 10.94% total PM missingness. A predeclared 15% station cutoff retains it, while masked targets and reliability flags keep the gap from becoming artificial training data.

## Outputs

Generated data and results are deliberately ignored by Git:

- `data/processed/dataset.npz`: feature tensors, masks, scalers, and split origins.
- `outputs/audit.json`: full day-one audit by station.
- `outputs/baselines/`: naive metrics and cached forecasts.
- `outputs/<model>/seed_<seed>/`: neural metrics, run metadata, and test forecasts.
- `checkpoints/`: best validation checkpoint for each run.

The network median daily temperature maximum occurs at recorded hour 12 in January and hour 13 in July. Following the proposal's diagnostic rule, the timestamps are treated as local civil clock time and the recorded wall-clock hour is used directly for calendar features. The grid remains complete through clock-change dates, so no timestamp is discarded or duplicated.

