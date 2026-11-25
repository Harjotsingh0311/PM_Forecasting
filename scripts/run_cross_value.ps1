$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$models = @("cnn", "lstm", "tcn")
$targets = @("pm25", "coarse")
$inputs = @("own", "both")
$seeds = @(42, 123)

foreach ($model in $models) {
    foreach ($target in $targets) {
        foreach ($inputSet in $inputs) {
            foreach ($seed in $seeds) {
                $name = "${model}_${target}_${inputSet}"
                $runMetadata = Join-Path $PSScriptRoot "..\outputs\cross_value\$name\seed_$seed\run.json"
                if (Test-Path -LiteralPath $runMetadata) {
                    Write-Host "Skipping completed run: $name seed $seed"
                }
                else {
                    & $python -m pm_forecasting.cli cross-value --model $model --target $target --inputs $inputSet --seed $seed
                }
            }
        }
    }
}

& $python -m pm_forecasting.cli cross-value-summary

