# All-TMD

All-TMD trains a car/bus/train classifier from a selected immutable training
dataset plus collector calibration sessions, then evaluates the fitted model
against collector holdout sessions. The first adapters support US-TMD and
NOR-TMD; the shared pipeline is designed so another source can be added with
one `TrainingDatasetAdapter` implementation.

## Configure and run

Copy the examples and set the host data directory:

```powershell
Copy-Item trials.json.example trials.json
Copy-Item .env.example .env
.\scripts\run-trials.ps1
```

`run-trials.ps1` builds the Docker image once, then runs training-dataset
ingestion, collector ingestion, feature extraction, and training for every
object in `trials.json`. It stops on the first failing stage.

Each trial object is hashed independently using canonical JSON. Fields in
`model.config.yaml` deliberately do not contribute to the hash. Outputs are
written beneath:

```text
all-tmd-work/<trial-sha256>/
|-- events/
|   |-- <train_dataset>/part-*.parquet
|   `-- collector/part-*.parquet
|-- features/
|   |-- <train_dataset>/part-*.parquet
|   `-- collector/part-*.parquet
|-- reports/<train_dataset>/
|   |-- metrics.json
|   |-- model.joblib
|   `-- optuna-trials.csv
|-- splits/<train_dataset>.json
`-- trial.json
```

## Incremental behavior

Immutable training event and feature datasets receive a `_SUCCESS` marker and
are skipped on later runs. Collector event and feature directories contain a
`checkpoint.json` array of processed session IDs. Newly downloaded collector
sessions append new Parquet parts without rebuilding existing sessions.

If a fixed training ingestion or feature build stops before writing `_SUCCESS`,
the next run removes that incomplete directory and rebuilds it. Collector
checkpoints are reconciled with existing Parquet parts after interruption.

There is intentionally no overwrite option. To force a rebuild, remove the
specific hashed output directory after confirming its path.

## Data sources

Container paths come from `model.config.yaml` and are resolved inside the
directory mounted at `/data`.

- `sources.us-tmd.input_dir` is scanned with its configured CSV globs.
- `sources.nor-tmd.input_dir` may be either one NOR-TMD CSV file or a directory
  recursively containing CSV files.
- `sources.collector.input_dir` contains collector JSON or JSON.GZ session
  exports and optional `.metadata.json` sidecars.

All adapters emit the same normalized event columns. To add another immutable
training source, add its source entry to `model.config.yaml` and implement a
concrete `TrainingDatasetAdapter` with a matching `dataset_name` in
`all_tmd/ingest.py`.

## Features and training

For every configured sensor, the minimum number of samples in a window is:

```text
default_window_seconds * minimum_sampling_rate[sensor]
```

Windows missing the required count for any configured sensor are discarded.
Feature columns use vector magnitude for accelerometer, gyroscope, and
magnetometer, and the scalar pressure value for pressure.

Collector groups are stratified into calibration and holdout sets according to
`training.calibration_fraction`. Optuna folds train on all source rows plus the
non-validation collector calibration groups. The final model trains on all
source rows plus all collector calibration rows; only collector holdout groups
are used for final reported evaluation.

MLflow is available at `http://localhost:5002` after:

```powershell
docker compose --profile mlflow up mlflow
```

## Direct commands

The trial index defaults to zero:

```powershell
python -m all_tmd.cli --trial-index 0 ingest-train-dataset
python -m all_tmd.cli --trial-index 0 ingest-collector
python -m all_tmd.cli --trial-index 0 features
python -m all_tmd.cli --trial-index 0 train
```

## Tests

```powershell
pytest
```
