$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$models = @("cnn", "lstm", "tcn", "ebamtn", "cpmtcn", "cpmtcn_no_exchange", "cpmtcn_soft")
$seeds = @(42, 123)

& $python -m pm_forecasting.cli audit
& $python -m pm_forecasting.cli prepare
& $python -m pytest -q
& $python -m pm_forecasting.cli baselines
& $python -m pm_forecasting.cli analyze

foreach ($classicalModel in @("ridge", "lightgbm")) {
    $classicalRun = Join-Path $PSScriptRoot "..\outputs\$classicalModel\seed_42\run.json"
    $classicalEfficiency = Join-Path $PSScriptRoot "..\outputs\$classicalModel\seed_42\efficiency.json"
    if ((-not (Test-Path -LiteralPath $classicalRun)) -or (-not (Test-Path -LiteralPath $classicalEfficiency))) {
        & $python -m pm_forecasting.cli classical --model $classicalModel --seed 42
    }
}

$chronosRun = Join-Path $PSScriptRoot "..\outputs\chronos2\seed_0\run.json"
if (-not (Test-Path -LiteralPath $chronosRun)) {
    & $python -m pm_forecasting.cli chronos2 --batch-size 16 --chunk-size 128
}

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

& $python -m pm_forecasting.cli efficiency
& $python -m pm_forecasting.cli significance
& $python -m pm_forecasting.cli report

