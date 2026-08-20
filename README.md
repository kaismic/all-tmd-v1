# All-TMD

All-TMD trains a transport mode classifier from a selected immutable training
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

## Run trials on AWS EC2

The AWS runner is intended for occasional, long CPU sweeps that should continue
without keeping the local computer available. It deploys one On-Demand
`c7i.4xlarge` worker in `ap-southeast-2`, an encrypted 200 GiB `gp3` data
volume, and a private encrypted S3 bucket. The worker has no inbound security
group rules; administration and on-demand MLflow port forwarding use Systems
Manager.

The EC2 data volume is mounted at `/mnt/all-tmd-data` on the host and at `/data`
inside the existing Compose services. Immutable training inputs are copied from
the ML S3 bucket, while confirmed collector sessions are downloaded directly
from the collector backend's S3 bucket at the start of every run. Collector
downloads, active events, features, and checkpoints remain on EBS for
incremental reuse. Each run writes its own MLflow SQLite database and artifacts
directly to isolated run storage, without a tracking server. Every run uploads
its reports, models, splits, configuration, logs, resource-usage report, and
run-specific MLflow state to S3 before stopping the worker.
The worker also stops after pipeline failure, once diagnostics are uploaded.

### Prerequisites and deployment

Install the AWS CLI and Session Manager plugin, authenticate an AWS profile,
and commit and push the exact All-TMD code that will run. The remote worker
checks out the full Git SHA recorded in each run bundle. The default preparation
command refuses a dirty worktree to prevent a local-only code revision from
being mistaken for the remote revision. Both `prepare-run.ps1` and
`start-run.ps1` verify that the exact commit can be fetched from the recorded
repository, and stop with a push-first error before starting EC2 when it cannot.

Deploy the collector backend first, then deploy this stack with the email
address that should receive the USD 50 monthly EC2 budget alert. The deploy
script reads the collector bucket and table from the `transport-data-collector`
CloudFormation stack; use `-CollectorStackName` if it has a different name:

```powershell
aws login
.\scripts\aws\deploy.ps1 -BudgetEmail you@example.com
```

Re-run this deployment command once after upgrading an existing All-TMD stack;
that applies the collector read permissions to its worker role. The script
temporarily starts a stopped worker for validation and stops it again afterward.

The deployment validates Docker, Compose, the EBS mount, and Systems Manager,
then stops the initialized worker unless `-LeaveRunning` is supplied. The stack
uses a generated bucket name by default; pass `-BucketName` when a specific
globally unique name is required. `-Profile` and `-Region` are accepted by all
AWS scripts.

If the ntfy topic requires an access token, store it as an SSM SecureString.
The token is never included in S3 run bundles or uploaded from the local `.env`:

```powershell
.\scripts\aws\set-ntfy-token.ps1
```

Upload the immutable training inputs once. `aws s3 sync` performs multipart
transfer for the large NOR-TMD source and does not delete remote objects.
Collector sessions are not uploaded through this command:

```powershell
.\scripts\aws\upload-inputs.ps1 -DataDir D:\tmd-data
```

At the start of each EC2 run, the worker queries the collector backend's
`received-sync-index`, downloads newly confirmed `participant_###` payloads
directly from its `raw/` S3 prefix, and records a persistent EBS checkpoint.
Uploads confirmed after that startup sync are intentionally picked up by the
next run, so a trial always trains against a stable collector snapshot.

### Smoke and full runs

Prepare a smoke bundle first. It contains only the first generated trial and
sets `training.optuna_trials` to `1`; the canonical parameter file is not
modified:

```powershell
.\scripts\aws\prepare-run.ps1 -Mode Smoke -NtfyTopic your-topic
.\scripts\aws\start-run.ps1 -RunId <printed-smoke-run-id>
.\scripts\aws\status.ps1 -RunId <printed-smoke-run-id>
```

After the smoke run succeeds, prepare the full bundle.

```powershell
.\scripts\aws\prepare-run.ps1 -Mode Full -NtfyTopic your-topic
.\scripts\aws\start-run.ps1 -RunId <printed-full-run-id>
```

`start-run.ps1` returns after installing and starting a one-shot systemd
service. The service continues when the Session Manager connection or local
computer closes. Obtain a status and recent journald lines at any time with
`status.ps1`. The EC2 trial runner does not start an MLflow server. To inspect an
unusually long active run, start one temporarily and open a private
port-forwarding session at `http://localhost:5002`:

```powershell
.\scripts\aws\port-forward-mlflow.ps1 -RunId <active-run-id>
```

The command starts the server through Systems Manager, keeps it available for
the forwarding session, and stops it when the session closes. Training keeps
writing directly to the run's SQLite database, so MLflow recording never
depends on the UI server. The MLflow run is created before Optuna begins, but
intermediate Optuna fold and trial scores are currently reported only through
`status.ps1` and journald; MLflow metrics and artifacts are logged when that
training step finishes.

The launcher waits for the worker to become available through Systems Manager,
which is the service used to install the run. If the EC2 instance is stopped
while it is starting, the launcher exits immediately with the observed EC2
state instead of waiting for the generic EC2 status-check timeout.

Download a completed run from its isolated S3 results prefix:

```powershell
.\scripts\aws\download-results.ps1 `
  -RunId <run-id> `
  -Destination .\aws-results\<run-id>
```

Print the collector payload snapshot captured at the start of a downloaded run:

```powershell
python .\scripts\report-collector-snapshot.py <run-id>
```

The report includes the number of sessions, total sample and uploaded-duration
figures by transport mode, and uploaded duration by participant and mode.
Durations use `HH:MM:SS`, including zero-valued fields (for example,
`52:35:00`). Pass `--latex` to print all three tables as left-aligned LaTeX
`tabular` environments with captions and with special characters in labels and
values escaped:

```powershell
python .\scripts\report-collector-snapshot.py <run-id> --latex
```

New EC2 runs upload `run/collector-snapshot.json`, containing the
sorted session IDs and the session metadata needed to reproduce this report
without access to EBS or the collector backend. The manifest also records its
RunId, capture time, source checkpoint, session count, and SHA-256 digest of the
session-ID set.

Runs created before this manifest was introduced can still be reported when
the same collector payload sidecars remain locally. Their session membership is
reconstructed from the uploaded `run/run.log`; the sidecars supply mode,
participant, sample count, and duration metadata. The command reads
`ALL_TMD_DATA_DIR` from the environment or `.env`, or accepts an explicit path:

```powershell
python .\scripts\report-collector-snapshot.py `
  <legacy-run-id> `
  --sessions-dir D:\tmd-data\downloaded_sessions
```

This is the raw payload snapshot fixed at sweep startup. A trial can use fewer
sessions after duration and feature-window quality filtering. Each MLflow run's
`collector_session_count` and `collector_session_digest` identify that trial's
effective collector feature-frame membership.

Find the best trial in one downloaded run by a collector-holdout selector
metric and display that metric together with recall for every transport mode:

```powershell
python .\scripts\show-best-trial.py `
  collector_holdout.balanced_accuracy `
  <run-id>
```

The other accepted selectors are `collector_holdout.accuracy`,
`collector_holdout.macro_f1`, and
`collector_holdout.minimum_class_recall`. The script reads canonical trial reports beneath
`aws-results/<run-id>/work` and excludes duplicate MLflow artifact copies.

Export the top three trials by collector-holdout macro F1, accuracy, and
balanced accuracy across every downloaded run as three LaTeX tables. Each
table also includes every transport mode's F1 score:

```powershell
python .\scripts\export-best-f1-table.py
```

The tables use MLflow run IDs, sort transport-mode columns by name, and print
four decimal places. Each transport-mode cell contains `F1-score, accuracy`,
where the mode-specific accuracy is its classification-report recall (the share
of that true mode classified correctly). Trials are ranked independently for
each table. Duplicate MLflow run IDs present in multiple downloads are included
once. Use `--limit <count>` or `--precision <digits>` to override the defaults.

Export collector-holdout recall for every trial and transport mode in one
downloaded run as a LaTeX `tabular` table:

```powershell
python .\scripts\export-recall-table.py <run-id>
```

Rows follow the trial index, transport-mode columns are sorted by name, and
LaTeX-special characters in run and mode names are escaped. Values use four
decimal places by default; pass `--precision <digits>` to change this. If a
trial does not contain a mode found in another trial, its table cell contains
`--`.

Compare the collector-holdout confusion matrices across all downloaded trial
results by true transport label, limiting the output to the top-ranked trials:

```powershell
python .\scripts\compare-confusion-matrices.py bus 10
```

The script normalizes the selected true-label row, ranks trials by its diagonal
(correct-prediction) value, and prints the MLflow trial name, full trial hash,
normalized predicted-value array with its label order, and support. MLflow's
duplicate artifact copies are excluded from the comparison.

View the downloaded MLflow database and artifacts locally with Docker:

```powershell
.\scripts\aws\view-results.ps1 -RunId <run-id>
```

The viewer starts the repository's pinned MLflow image in the background,
waits for it to become healthy, and opens `http://127.0.0.1:5003`. This uses a
different default host port from the local Docker trial UI at
`http://localhost:5002`, so both can run at the same time. Omit
`-RunId` to select the most recently modified run beneath `aws-results` that
contains an MLflow database. Use `-ResultsRoot` for a different download root,
`-LocalPort` if port 5003 is occupied, or `-NoBrowser` to suppress automatic
browser launch. The downloaded database and artifact directory are mounted
directly, so the EC2 worker does not need to be running.

Stop the local viewer when finished:

```powershell
.\scripts\aws\view-results.ps1 -Stop
```

Pass the same `-LocalPort` to `-Stop` when an alternate port was used. The
viewer container is removed automatically; the downloaded results remain.

The uploaded `run/run-summary.json` records the exit code, duration, vCPU and
memory allocation. `run/resource-usage.txt` records GNU `time -v` measurements,
including peak resident memory and CPU utilization. EC2 detailed monitoring
and EBS CloudWatch metrics provide instance CPU and volume throughput history.

Use `stop-worker.ps1` if a debugging run was prepared with `-NoAutoStop`. A
stopped worker incurs no EC2 compute charge, but EBS storage remains billable.
If another sweep is not expected within 30 days, archive the stack explicitly:

```powershell
.\scripts\aws\archive-stack.ps1 -ConfirmArchive
```

Archiving deletes the worker and its dedicated network, creates a final EBS
snapshot through CloudFormation, and retains the S3 bucket. Snapshots and S3
objects remain billable until deliberately deleted. The CloudFormation outputs
identify every persistent resource.

### Generate trial configurations

`trial-parameters.json` contains a complete `default` trial and zero or more
named `dimensions`. The generator chooses one option from each dimension and
writes their Cartesian product to `trials.json`. Options in the first
dimension vary slowest. The example independently varies the sensor set, the
accelerometer feature set, and the paired window/step values.

Each trial may include an optional top-level `run_name` string. It is display
metadata and does not affect either configuration hash. A dimension can vary
it with `set` alongside the behavior it describes, for example:

```json
{
  "set": {
    "run_name": "hierarchical-domain-1",
    "training.weighting_strategy": "hierarchical",
    "training.collector_domain_weight": 1.0
  }
}
```

Each option supports these operations:

- `set` maps dotted trial paths to replacement JSON values. Put related paths
  in the same option to keep them paired, as the example does for
  `features.default_window_seconds` and `features.default_step_seconds`.
- `pick` maps a dotted path for either a JSON object or a string array to the
  keys or values to retain, in the requested order. The example uses it to
  select sensor subsets and accelerometer feature subsets while defining the
  available values only once in `default`. Array values must match the
  corresponding values in `default` exactly.

Every option in one dimension must modify the same paths. Separate dimensions
cannot modify overlapping paths except when nested paths both use `pick`, such
as `features.sensors` and `features.sensors.accelerometer`. Nested picks are
applied from the deepest path outward, so a feature set can be selected before
its sensor is retained or removed. All paths and selected keys or values must
already exist in `default`; these checks catch misspellings before
`trials.json` is replaced.

The generator removes identical resulting trials while preserving their
first-seen order. This prevents the feature choices for a sensor excluded by a
sensor-set option from producing redundant training runs. An empty
`dimensions` array generates one copy of the default trial.

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

Each trial has two hashes. The feature-cache hash uses canonical JSON after
removing the top-level `training` field; neither that field nor
`model.config.yaml` contributes, so training-only variants reuse ingestion and
features. The full trial hash includes training settings and gives every model,
report, split, and trial snapshot an isolated path. The most recently executed
trial also remains at the feature-cache root as `trial.json` for compatibility.
Outputs are written beneath:

```text
all-tmd-work/<feature-cache-sha256>/
|-- events/
|   |-- <train_dataset>/part-*.parquet
|   `-- collector/part-*.parquet
|-- features/
|   |-- <train_dataset>/part-*.parquet
|   `-- collector/part-*.parquet
|-- reports/<train_dataset>/<full-trial-sha256>/
|   |-- metrics.json
|   |-- model.joblib
|   |-- optuna-trials.csv
|   |-- nested-optuna-trials.csv
|   `-- trial.json
|-- splits/<train_dataset>/<full-trial-sha256>.json
`-- trial.json
```

## Incremental behavior

Immutable training event and feature datasets receive a `_SUCCESS` marker and
are skipped on later runs. Collector event and feature directories contain a
`checkpoint.json` array of processed session IDs. Newly downloaded collector
sessions append new Parquet parts without rebuilding existing sessions.

Feature directories also contain `feature-policy.json`, which records the
settings that affect their contents. Collector policies include collector
sampling and continuity settings; training-source policies do not. Missing,
legacy, corrupt, or changed policy metadata causes only that source's feature
directory to be rebuilt automatically; event data remains reusable.

Long-running ingestion and feature operations emit flushed, timestamped
progress. Collector scans report every input file and classify it as ingested,
already processed, ignored, or duration-filtered, together with cumulative
counters. Feature extraction reports checkpoint reconciliation, event-session
scanning, input-part bucketing, bucket processing, and session processing. A
10-second heartbeat is emitted while an individual session takes unusually
long to window, so a quiet terminal does not look like a frozen pipeline.

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

For every configured sensor, the collector minimum number of samples in a
window is:

```text
default_window_seconds * collector_minimum_sampling_rate[sensor]
```

Collector windows missing the required count for any configured sensor are
discarded. When `dataset.collector_max_sample_interval_ms` is not `null`, each
configured sensor must also cover the complete collector window without a gap
greater than that value. The check includes the window-start-to-first-sample
gap, consecutive sample gaps, and the last-sample-to-window-end gap. A bad
window is discarded without discarding other windows from the same session.

Training-source windows do not use the collector sampling-rate or continuity
thresholds. They require at least one valid value for every configured sensor
so feature aggregation remains defined. Sparse source windows may consequently
produce degenerate statistics such as zero variance or range.

Collector exports are emitted on accelerometer callbacks and carry the latest
available values for the other sensors. Consequently, collector continuity is
evaluated from the timestamps represented in the export and cannot detect how
long a cached gyroscope, magnetometer, or pressure value has remained
unchanged.

The generic `minimum_sampling_rate` and
`dataset.maximum_sample_interval_ms` keys are no longer accepted. Rename them
to `collector_minimum_sampling_rate` and
`dataset.collector_max_sample_interval_ms`, respectively. Set the maximum
interval to `null` to disable collector continuity filtering while retaining
collector minimum-count filtering.

Feature columns use vector magnitude for accelerometer, gyroscope, and
magnetometer, and the scalar pressure value for pressure.

Time-domain aggregations include minimum, maximum, mean, range, variance,
standard deviation, kurtosis, interquartile range, mean absolute deviation, and
the existing delta features. Optional motion and frequency aggregations are
`mean_absolute_jerk`, `jerk_standard_deviation`, `spectral_energy`,
`dominant_frequency_hz`, `spectral_entropy`, and `mean_axis_correlation`.
Frequency features linearly resample each irregular sensor series at its median
positive interval, remove its mean, and calculate the real-valued spectrum.

Set `features.context_windows_seconds` to extract the configured sensor features
over multiple trailing contexts in every outer window. The list must contain
unique positive whole seconds, cannot exceed `default_window_seconds`, and must
include that default. A 60-second outer window with `[10, 30, 60]` produces
columns such as `accelerometer#spectral_energy@10s` and
`accelerometer#spectral_energy@60s`. Omitting the field preserves the original
unsuffixed single-context column names. Collector sampling and continuity
requirements are checked independently for every context.

Collector groups are split into calibration and holdout sets independently for
each configured transport mode. `training.calibration_fraction` can be either a
single fraction applied to every mode or an object with exactly one fraction
for every key in `labels`:

```json
"calibration_fraction": {
  "bus": 0.8,
  "car": 0.4,
  "train": 0.4
}
```

Fractions apply to whole collector groups, not individual feature rows. For
each mode, the calibration group count is the floor of the group count times
its fraction. The result is clamped to leave at least one group in both
calibration and holdout, so every configured mode requires at least two
collector groups. The split is deterministic for the trial's `random_seed`.

Optuna folds train on all source rows plus the non-validation collector
calibration groups. The final model trains on all source rows plus all collector
calibration rows; only collector holdout groups are used for final reported
evaluation. The effective per-mode fractions and group counts are recorded in
the split manifest and the fractions are logged as MLflow parameters.

When a class has only two collector groups, one group remains in calibration
and one remains in holdout. Grouped cross-validation still evaluates every
calibration group exactly once, and Optuna is scored from the combined
out-of-fold predictions rather than requiring every class in every fold.

Because calibration fractions are under the trial's top-level `training`
field, changing them does not change the feature-cache hash. Existing ingested
events and extracted features are reused, while the full trial hash gives the
new split, model, reports, and MLflow run their own identity.

### Participant-independent evaluation and balancing

The legacy default `training.evaluation_strategy` is `session_holdout`, with the
session-group behavior described above. Set it to `participant_nested_cv` when
the model must generalize to people absent from training. Every
collector participant becomes one outer evaluation fold. Optuna tuning for that
fold sees only the other participants, using stratified participant-grouped
inner folds. The pooled outer predictions are stored under the established
`collector_holdout` metrics key so the comparison scripts remain compatible.
After unbiased evaluation, a separate final study tunes on participant-grouped
folds and the deployable model trains on all collector rows plus the immutable
source dataset.

Nested evaluation performs one Optuna study per participant plus a final study,
so it is substantially more expensive than the session holdout. Use a small
`optuna_trials` value for smoke and ablation runs before a final model search.
`training.participant_inner_folds` defaults to five and is capped at the number
of available training participants. Every configured mode must occur for at
least two collector participants, and at least three participants are required
overall.

Set `training.weighting_strategy` to `hierarchical` to prevent long recordings
from dominating. Within each domain and class, total weight is divided equally
between participants, then sessions, then windows. `collector_domain_weight`
controls the collector domain's total contribution relative to the immutable
source domain; it defaults to `2.0`. The alternative `class_balanced` strategy
retains inverse-frequency row weights. Random forest estimator-level
`class_weight` is disabled because it would multiply the pipeline's sample
weights and apply the correction twice.

`training.duration_balancing` accepts `none` or `smallest_mode`. The latter is
an ablation, not an evaluation filter: for every training fold it keeps all
source rows but selects whole collector sessions until each mode approximates
the smallest available mode duration. Selection cycles deterministically across
participants before taking another session from the same participant. Outer
evaluation rows are never removed, and the split report records the target,
selected session counts, and selected usable durations.

Optuna can optimize `training.selection_metric` as either `macro_f1` (the
default) or `minimum_class_recall`. The latter directly targets the weakest
mode. Every evaluation now also reports participant and phone-position
breakdowns, mean-probability session-level predictions, and
`minimum_class_recall`. Set `training.bootstrap_iterations` above zero to add
deterministic 95% intervals that resample whole participants rather than
correlated windows.

The version-controlled `recommended-trial-plans/` directory contains the
generated staged experiments for the current collector snapshot. Every plan
uses descriptive `run_name` values for quick identification in logs, MLflow,
and downloaded result summaries. Generate any plan independently without
replacing the canonical parameter file:

```powershell
python .\scripts\generate-trials.py `
  --parameters .\recommended-trial-plans\02-weighting-ablation.trial-parameters.json `
  --output .\recommended-trial-plans\02-weighting-ablation.trials.json
```

For local, non-AWS execution, MLflow is available at
`http://localhost:5002`. Both local trial runners start it automatically; to
start it without running trials, use:

```powershell
  docker compose --profile mlflow up -d --wait mlflow
```

Training containers connect to `http://mlflow:5002` on the Compose network.
The server stores metadata in `/data/all-tmd-work/mlflow.db` and proxies
artifacts into `/data/all-tmd-work/mlartifacts`, both beneath the configured
`ALL_TMD_DATA_DIR` host directory.

AWS bundles instead set `mlflow.tracking_uri` to
`sqlite:////mlflow-data/mlflow.db` and `mlflow.artifact_location` to
`file:///mlflow-data/mlartifacts`. The training client writes directly to these
run-specific paths, and the cloud worker uploads them only after the sweep
finishes. The optional `mlflow.artifact_location` setting is left as `null` for
the local server-backed workflow.

New runs appear in the stable `ALL-TMD` experiment. Collector downloads do not
rename or version the experiment. Instead, every run records the
`collector_session_digest` parameter, a SHA-256 fingerprint of its sorted
unique collector session IDs, plus `collector_session_count`. These fields make
runs that use the same collector snapshot directly searchable and comparable
without storing local download state in `model.config.yaml`.

Without `run_name`, run names use
`<train-dataset>-<8-character-full-trial-hash>-<UTC-start-time>-<collector-session-count>`,
for example `nor-tmd-a1b2c3d4-20260812T034512Z-42`. When configured, the custom
value is prepended, producing
`<run-name>-<train-dataset>-<hash>-<time>-<count>`. The same behavior is used by
local Docker and AWS EC2 trials. The compact UTC timestamp distinguishes
repeated runs of the same trial, while the final value makes the collector
snapshot size visible in the MLflow run list. The custom value is also logged
as `configured_run_name`, written to `metrics.json`, shown in command progress,
and used by the downloaded-result comparison scripts.

Run parameters also expose the complete configured feature set. The existing
`sensors` parameter provides the sensor names, `feature_names` provides every
fully qualified feature such as `accelerometer#standard_deviation`, and each
configured sensor has its own comma-separated parameter such as
`features.accelerometer=mean,standard_deviation,range`. Sensors omitted from a
trial do not create `features.<sensor>` parameters.

Runs also record `collector_max_sample_interval_ms` and one
`collector_minimum_sampling_rate.<sensor>` parameter per configured sensor so
collector quality filtering remains reproducible in MLflow.

The **Datasets** section of each run records three native MLflow inputs: source
training features, collector calibration features, and collector holdout
features. Each input has a deterministic content digest covering its labels,
group and session IDs, window boundaries, and configured feature values. The
digest is independent of row order but changes when any tracked content
changes. It uses the first 36 hexadecimal characters of the SHA-256 fingerprint
to fit MLflow's dataset-digest field. Dataset contexts are `training`,
`calibration`, and `evaluation`, respectively.

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
