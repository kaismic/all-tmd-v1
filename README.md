# All-TMD

All-TMD trains a car/bus/train classifier from a selected immutable training
dataset plus collector calibration sessions, then evaluates the fitted model
against collector holdout sessions. The first adapters support US-TMD and
NOR-TMD; the shared pipeline is designed so another source can be added with
one `TrainingDatasetAdapter` implementation.

## Configure and run

Copy the examples, generate the Cartesian product of trials, and set the host
data directory:

```powershell
Copy-Item trial-parameters.json.example trial-parameters.json
Copy-Item .env.example .env
python .\scripts\generate-trials.py
.\scripts\run-trials.ps1
```

On Linux or macOS, use the Bash runner instead. It requires either `python3` or
`jq` to read `trials.json`:

```bash
cp trial-parameters.json.example trial-parameters.json
cp .env.example .env
python3 ./scripts/generate-trials.py
bash ./scripts/run-trials.sh
```

Both runners build the Docker image once, start the MLflow service and wait for
it to become healthy, then run training-dataset ingestion, collector ingestion,
feature extraction, and training for every object in `trials.json`. They stop
on the first failing stage and leave MLflow running so its UI remains
available.

### Generate trial configurations

`trial-parameters.json` contains a complete `default` trial and zero or more
named `dimensions`. The generator chooses one option from each dimension and
writes their Cartesian product to `trials.json`. Options in the first
dimension vary slowest, so the example produces the four window choices for
the first sensor set before moving to the next sensor set.

Each option supports these operations:

- `set` maps dotted trial paths to replacement JSON values. Put related paths
  in the same option to keep them paired, as the example does for
  `features.default_window_seconds` and `features.default_step_seconds`.
- `pick` maps a dotted path for a JSON object to the list of keys to retain.
  The example uses it to select sensor subsets while defining each sensor's
  aggregation list only once in `default`.

Every option in one dimension must modify the same paths, and separate
dimensions cannot modify overlapping paths. All paths and `pick` keys must
already exist in `default`; these checks catch misspellings before
`trials.json` is replaced. An empty `dimensions` array generates one copy of
the default trial.

For example, another independent dimension can vary datasets:

```json
{
  "name": "training-dataset",
  "options": [
    { "set": { "train_dataset": "us-tmd" } },
    { "set": { "train_dataset": "nor-tmd" } }
  ]
}
```

Use alternate paths when needed:

```powershell
python .\scripts\generate-trials.py `
  --parameters custom-parameters.json `
  --output custom-trials.json
```

## Notifications

Set `NTFY_TOPIC` in `.env` to enable ntfy notifications. `NTFY_EVENTS` controls
which completed or failed pipeline events publish a notification:

```dotenv
# Notify once when the trial runner exits. A successful event means all trials ran.
NTFY_EVENTS=all-trials

# Notify after every training step (once per trial).
NTFY_EVENTS=train

# Notify after both ingestion tasks and feature extraction in every trial.
NTFY_EVENTS=ingest,features
```

The default, `steps`, preserves the original behavior and selects every
per-trial task: `ingest-train-dataset`, `ingest-collector`, `features`, and
`train`. The `ingest` alias selects both ingestion tasks. `all` selects every
task plus `all-trials`; `none` disables all events without clearing
`NTFY_TOPIC`. Values are case-insensitive and may be separated by commas or
spaces.

The `all-trials` event is sent once as the trial runner exits. It reports success
after the script completes all configured trials, or failure if the script
stops early. Notification delivery is best-effort and does not change a
pipeline task's exit status.

Each trial object is hashed independently using canonical JSON after removing
its top-level `training` field. Neither that field (including all its
subproperties) nor fields in `model.config.yaml` contribute to the hash, so
trials that differ only in training settings reuse the same ingested events
and extracted features. The full current trial remains recorded in
`trial.json`. Outputs are written beneath:

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

When a class has only two collector groups, one group remains in calibration
and one remains in holdout. Grouped cross-validation still evaluates every
calibration group exactly once, and Optuna is scored from the combined
out-of-fold predictions rather than requiring every class in every fold.

MLflow is available at `http://localhost:5002`. Both trial runners start it
automatically; to start it without running trials, use:

```powershell
docker compose --profile mlflow up -d --wait mlflow
```

Training containers connect to `http://mlflow:5002` on the Compose network.
The server stores metadata in `/data/all-tmd-work/mlflow.db` and proxies
artifacts into `/data/all-tmd-work/mlartifacts`, both beneath the configured
`ALL_TMD_DATA_DIR` host directory.

New runs appear in the stable `ALL-TMD` experiment. Collector downloads do not
rename or version the experiment. Instead, every run records the
`collector_session_digest` parameter, a SHA-256 fingerprint of its sorted
unique collector session IDs, plus `collector_session_count`. These fields make
runs that use the same collector snapshot directly searchable and comparable
without storing local download state in `model.config.yaml`.

The **Datasets** section of each run records three native MLflow inputs: source
training features, collector calibration features, and collector holdout
features. Each input has a deterministic content digest covering its labels,
group and session IDs, window boundaries, and configured feature values. The
digest is independent of row order but changes when any tracked content
changes. Dataset contexts are `training`, `calibration`, and `evaluation`,
respectively.

Each new run also records `metrics.json`, the fitted model, Optuna trials, and
the split manifest in the **Artifacts** tab. The `evaluation` artifact
directory contains raw-count and row-normalized collector-holdout
confusion-matrix images. Matrix rows are actual labels and columns are
predicted labels.

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
