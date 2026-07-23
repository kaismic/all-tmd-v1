param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$trialsPath = Join-Path $projectRoot "trials.json"
if (-not (Test-Path -LiteralPath $trialsPath -PathType Leaf)) {
    throw "trials.json was not found. Copy trials.json.example to trials.json and edit it before running trials."
}

$trials = @(Get-Content -LiteralPath $trialsPath -Raw | ConvertFrom-Json)
if ($null -eq $trials -or $trials.Count -eq 0) {
    throw "trials.json must contain at least one trial object."
}

Push-Location $projectRoot
$previousTrialIndex = $env:ALL_TMD_TRIAL_INDEX
$runStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$runExitCode = 1
$runFailure = $null
try {
    docker compose build
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose build failed with exit code $LASTEXITCODE."
    }

    docker compose --profile mlflow up -d --wait mlflow
    if ($LASTEXITCODE -ne 0) {
        throw "MLflow failed to start with exit code $LASTEXITCODE."
    }

    for ($index = 0; $index -lt $trials.Count; $index++) {
        $env:ALL_TMD_TRIAL_INDEX = [string]$index
        Write-Host "Running All-TMD trial $($index + 1)/$($trials.Count) (index $index)"

        docker compose --profile ingest run --rm ingest-train-dataset
        if ($LASTEXITCODE -ne 0) { throw "Training dataset ingestion failed for trial index $index." }

        docker compose --profile ingest run --rm ingest-collector
        if ($LASTEXITCODE -ne 0) { throw "Collector ingestion failed for trial index $index." }

        docker compose --profile features run --rm features
        if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed for trial index $index." }

        docker compose --profile train run --rm train
        if ($LASTEXITCODE -ne 0) { throw "Training failed for trial index $index." }
    }
    $runExitCode = 0
}
catch {
    $runFailure = $_
}
finally {
    $runStopwatch.Stop()
    docker compose --profile notifications run --rm --no-deps notify `
        run-trials `
        --event all-trials `
        --exit-code $runExitCode `
        --duration-seconds $runStopwatch.Elapsed.TotalSeconds
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The run-level ntfy notification command failed with exit code $LASTEXITCODE."
    }

    if ($null -eq $previousTrialIndex) {
        Remove-Item Env:ALL_TMD_TRIAL_INDEX -ErrorAction SilentlyContinue
    }
    else {
        $env:ALL_TMD_TRIAL_INDEX = $previousTrialIndex
    }
    Pop-Location
}

if ($null -ne $runFailure) {
    throw $runFailure
}
