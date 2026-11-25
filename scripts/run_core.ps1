$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$models = @("cnn", "lstm", "tcn", "ebamtn", "cpmtcn", "cpmtcn_no_exchange")
$seeds = @(42, 123)

& $python -m pm_forecasting.cli audit
& $python -m pm_forecasting.cli prepare
& $python -m pytest -q
& $python -m pm_forecasting.cli baselines
& $python -m pm_forecasting.cli analyze

foreach ($model in $models) {
    foreach ($seed in $seeds) {
        $runMetadata = Join-Path $PSScriptRoot "..\outputs\$model\seed_$seed\run.json"
        if (Test-Path -LiteralPath $runMetadata) {
            Write-Host "Skipping completed run: $model seed $seed"
        }
        else {
            & $python -m pm_forecasting.cli train --model $model --seed $seed
        }
    }
}

& $python -m pm_forecasting.cli report

