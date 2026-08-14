# All-TMD
The core ingestion, feature extraction and training pipeline should reference `us-tmd-v2`. The ingestion steps should reference both `nor-tmd-v2` and `us-tmd-v2`.

## Implementation Requirements / Core Differences from `us-tmd-v2`
The project code should utilise polymorphism, and must be extensible for another training dataset, and allow adding another dataset adaptation by simpling implementing another concrete implementation, not limited by the currently available datasets (US-TMD, NOR-TMD).

The majority of the configuration fields in `model.config.yaml` has been moved to `trials.json`. `trials.json` contains a list of trial configurations.

Phase has been removed from training steps. Instead, the entry points of this project should be `scripts/run-trials.ps1` on Windows and `scripts/run-trials.sh` on Linux or macOS. When either script is run, it should first run:

```
docker compose build
```

And iterate each trial objects in `trials.json` file, and for each object, run something similar like:

```
docker compose --profile ingest run --rm ingest-train-dataset
docker compose --profile ingest run --rm ingest-collector
docker compose --profile features run --rm features
docker compose --profile train run --rm train
```

I will dynamically edit the content of `trials.json` to manually tweak configurations and evaluate metrics heuristically with mlflow.

### Configuration Hash
Currently, in `us-tmd-v2` the configuration hash is created by appending all the fields in `model.config.yaml`. However, in `all-tmd-v1`, the configuration hash should be created for each objects in `trials.json` separately.

However, the `training` field and its subproperties in the trial objects should be excluded from configuration hash input. This is because they do not influence ingestion and feature extraction output.

### Raw Data Ingestion
This project must be able to ingest both US-TMD and NOR-TMD dataset, and produce outputs in a common format.
For example,
```
all-tmd-work/<config-hash>/events/us-tmd/part-000000.parquet
all-tmd-work/<config-hash>/events/us-tmd/part-000001.parquet
...
all-tmd-work/<config-hash>/events/nor-tmd/part-000000.parquet
all-tmd-work/<config-hash>/events/nor-tmd/part-000001.parquet
...
all-tmd-work/<config-hash>/events/collector/part-000000.parquet
all-tmd-work/<config-hash>/events/collector/part-000001.parquet
...
```

#### Raw data ingestion should be incremental:
- Training dataset (us-tmd, nor-tmd, etc.)
Running a trial should scan the `events/<training-dataset>` directory, and skip ingestion entirely if a `_SUCCESS` marker is present. This is because raw training datasets do not change.
- Collector dataset
Running a trial should generate a checkpoint file in `events/collector`, which contains a list of session ids that were ingested. When running a trial, read the checkpoint file, and skip session IDs already present in the path, and generate Parquet files for newly downloaded sessions.

Overwrite option is not required.

### Feature extraction
Here is an example directory structure
```
all-tmd-work/<config-hash>/features/us-tmd/part-000000.parquet
all-tmd-work/<config-hash>/features/us-tmd/part-000001.parquet
...
all-tmd-work/<config-hash>/features/nor-tmd/part-000000.parquet
all-tmd-work/<config-hash>/features/nor-tmd/part-000001.parquet
...
all-tmd-work/<config-hash>/features/collector/part-000000.parquet
all-tmd-work/<config-hash>/features/collector/part-000001.parquet
...
```

Feature extraction should also be incremental, similar to raw data ingestion.

Long-running collector scans and feature extraction stages should expose
timestamped progress. Collector output should report each file's outcome and
cumulative counters. Feature output should cover checkpoint reconciliation,
event-session discovery, Parquet bucketing, bucket processing, and individual
session windowing, including a periodic heartbeat for long sessions.

### Training & Testing
For each trial, the models should be trained by utilising both the dataset specified by `train_dataset` field in the `trials.json` objects and the collector dataset, just like `adapt` phase in `us-tmd-v2`.
The models should be tested/evaluated with the collector dataset.

### Reports
Here is an example directory structure
```
all-tmd-work/<config-hash>/reports/us-tmd/metrics.json
all-tmd-work/<config-hash>/reports/us-tmd/model.joblib
all-tmd-work/<config-hash>/reports/us-tmd/optuna-trials.csv
all-tmd-work/<config-hash>/reports/nor-tmd/metrics.json
all-tmd-work/<config-hash>/reports/nor-tmd/model.joblib
all-tmd-work/<config-hash>/reports/nor-tmd/optuna-trials.csv
...
```

### Calculating minimum number of samples per window
`minimum_samples_per_sensor` was removed from `model.config.yaml`, and instead a new field `collector_minimum_sampling_rate` was added. Its Hz values apply only to collector windows and calculate the minimum number of samples by using the following equation:

```
minimum_samples_per_window = default_window_seconds * collector_minimum_sampling_rate
```

Suppose a `model.config.yaml` with the following configuration:
```
collector_minimum_sampling_rate: # hz
  accelerometer: 30
  gyroscope: 4
  magnetometer: 2
  pressure: 2
...
```

Then the `minimum_samples_per_window` for accelerometer is:

```
minimum_samples_per_window = 10 * 30 = 300
```

### Per-window temporal continuity

Collector feature windows must also remain temporally continuous when
`dataset.collector_max_sample_interval_ms` is configured. For every configured
sensor, feature extraction checks the window boundaries and consecutive valid
sample timestamps. Only the affected collector window is excluded when a gap
exceeds the threshold; the remaining windows in that session stay eligible.
Training-source windows instead require one valid value per configured sensor
and do not use the collector quality thresholds.

Each feature directory records its effective settings in `feature-policy.json`.
Collector settings appear only in the collector policy, so later quality
changes rebuild collector features without invalidating training-source
features. Missing or changed policies still rebuild the affected cached
features while preserving normalized event data.

### Notes
For any other undefined implementation details, adapt the implementation from `us-tmd-v2`.
For example, `.gitignore`, `.dockerignore`, `.env`, `pyproject.toml`, `requirements.txt`, `docker-compose.yml`, `Dockerfile`, etc. file contents and configurations.

## AWS EC2 execution

Occasional full sweeps are supported by a dedicated CloudFormation stack in
Sydney. The stack provisions an On-Demand `c7i.4xlarge`, an encrypted 200 GiB
`gp3` data volume, an encrypted/versioned private S3 bucket, a no-ingress
security group, least-privilege worker role, Systems Manager access, detailed
EC2 monitoring, and a USD 50 monthly EC2 budget notification. S3 is retained
and the EBS volume is snapshotted when the stack is archived.

Each cloud run uses an immutable, non-secret S3 bundle containing the exact Git
commit, generated trials, model configuration, parameter source, checksums, and
notification parameter name. A smoke bundle reduces the first generated trial
to one Optuna evaluation; a full bundle preserves all generated trials. The
current parameter document produces eight full trials.

The worker downloads immutable training inputs from the ML bucket and queries
the collector backend's received-session index at each run startup to copy new
confirmed collector payloads directly from its S3 bucket to persistent EBS.
The sync is checkpointed and gives every trial sweep a stable collector
snapshot. The worker runs the existing Bash and Docker Compose entry points
under a one-shot systemd service. It retrieves the ntfy token from SSM
Parameter Store and records MLflow metadata and artifacts directly into
run-specific SQLite and filesystem storage, so no MLflow server consumes
capacity during normal trial execution. An on-demand server can be started
through the SSM port-forward command for an active run and is stopped when the
viewing session closes. The worker uploads run-specific reports/models/splits
plus MLflow and diagnostic state, and stops the EC2 instance after success or
failure. There is no public application API, model schema, or trial schema
change for cloud execution.
