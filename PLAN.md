# All-TMD
The core ingestion, feature extraction and training pipeline should be based on `us-tmd-v2`. The ingestion steps should reference both `nor-tmd-v2` and `us-tmd-v2`.

## Implementation Requirements / Core Differences from `us-tmd-v2`
The project code should utilise polymorphism, and must be extensible for another training dataset, and allow adding another dataset adaptation by simpling implementing another concrete implementation, not limited by US-TMD and NOR-TMD dataset.

The majority of the configuration fields in `model.config.yaml` has been moved to `trials.json`. `trial.json` contains a list of trial configurations.

Phase has been removed from training. Instead, the entry point of this project should be `scripts/run-trials.ps1`. When the script is run, it should first run:

```
docker compose build
```

And iterate each trial objects in `trial.json` file, and for each object, run something similar like:

```
docker compose --profile ingest run --rm ingest-train-dataset
docker compose --profile ingest run --rm ingest-collector
docker compose --profile features run --rm features
docker compose --profile train run --rm train
```

I will dynamically edit the content of `trials.json` to manually tweak configurations and evaluate metrics heuristically with mlflow.

### Configuration Hash
Currently, in `us-tmd-v2` the configuration hash is created by appending all the fields in `model.config.yaml`. However, in `all-tmd-v1`, the configuration hash should be created for each objects in `trials.json` separately.

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

### Training & Testing
The models should be trained with the dataset specified by `train_dataset` in the `trials.json` objects.
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
`minimum_samples_per_sensor` was removed from `model.config.yaml`, and instead a new field `minimum_sampling_rate` was added. You should utilise each hz values for each sensor to calculate the minimum number of samples per window by using the following equation:

```
minimum_samples_per_window = default_window_seconds * minimum_sampling_rate
```

Suppose a `model.config.yaml` with the following configuration:
```
minimum_sampling_rate: # hz
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

### Notes
For any other unspecified implementation details, adapt and copy the implementation from `us-tmd-v2`.
Let me know if you have any clarifications, questions, or concerns.