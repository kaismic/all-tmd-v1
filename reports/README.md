# Preliminary experiment metrics

Each CSV contains exactly these columns, in this order:

```text
experiment_name,run_id,transport_mode,support,precision,recall,f1,accuracy
```

| CSV | Runs | Rows |
| --- | ---: | ---: |
| [initial-sensor-window-size.csv](initial-sensor-window-size.csv) | 12 | 36 |
| [pressure-feature-four-transport-classes.csv](pressure-feature-four-transport-classes.csv) | 12 | 48 |
| [magnetometer-feature.csv](magnetometer-feature.csv) | 12 | 36 |
| [calibration-fraction.csv](calibration-fraction.csv) | 3 | 9 |

One row represents one run and transport mode on **collector holdout feature
windows**. `experiment_name` is the descriptive experiment name, and `run_id`
is the full MLflow run ID. `support` counts true windows of that mode.
`precision`, `recall`, and `f1` are the saved per-mode classification metrics.
`accuracy` is the overall multiclass holdout accuracy, repeated for each mode
in the run; it is not a per-mode accuracy.

Scores are fractions in [0, 1], without deliberate rounding. Files use UTF-8
and standard CSV quoting. Rows are ordered by recorded trial index, run ID,
and transport mode. All eight pressure-enabled runs retain the saved tram
rows with zero true support and zero precision, recall, and F1.

## Regenerate

From `ml/all-tmd-v1`, run the standard-library-only exporter:

```powershell
python .\scripts\export-preliminary-experiment-metrics.py
```

The local work directory defaults to `ALL_TMD_DATA_DIR/all-tmd-work`, using
the environment variable first, then the repository `.env`. If neither is
set, it uses this repository's `data/all-tmd-work`. On the machine used for
these exports, it is `D:/tmd-data/all-tmd-work`. Override input or output
locations explicitly when needed:

```powershell
python .\scripts\export-preliminary-experiment-metrics.py `
  --local-work-root D:/tmd-data/all-tmd-work `
  --results-root ./aws-results `
  --output-dir ./reports
```

The script reads original artifacts and recreates all four CSVs. It checks
all source inputs before replacing any output, and fails if expected runs or
metrics are missing. Existing CSVs are not inputs.

## Sources and selection

The initial sweep comes from the local `mlflow.db`, experiment
`ALL-TMD Transfer v2`, and its `mlartifacts/<experiment_id>/<run_id>/artifacts/metrics.json`
files. Only active, finished runs with recorded holdout macro F1 are selected;
this excludes the artifact-storage check. The database and source artifacts
are external to this Git repository.

The other experiments use these downloaded bundles under `aws-results`:

- Pressure: `20260809T055942Z-9c676a2c-full`.
- Magnetometer: `20260812T125950Z-a46cc6a0-full`.
- Calibration fraction: `20260814T114054Z-c11f9a1a-full`.

For each bundle, the script selects active, finished `ALL-TMD` runs whose
start times fall within the execution interval in `run/run-summary.json`.
It reads their metrics from
`mlflow/mlartifacts/<experiment_id>/<run_id>/artifacts/metrics.json`.
This excludes earlier runs copied into later databases and preserves all
three calibration trials despite their shared work-directory reports being
overwritten by the last trial.

The four exports contain 39 runs and 129 unique `(run_id, transport_mode)`
rows. Cohorts and holdout membership differ across configurations, so these
are not evaluations on one common test set.
