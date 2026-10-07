# Detailed preliminary experiment metrics

Extracted on 2026-10-07 from original per-run MLflow artifacts. Each CSV row is
one completed run and transport mode. The primary metrics describe **collector
holdout feature windows**. Calibration metrics occupy separate columns in the
same row; they are fitted-model training-set results, not held-out validation.

| CSV | Runs | Rows | Columns |
| --- | ---: | ---: | ---: |
| [initial-sensor-window-size.csv](initial-sensor-window-size.csv) | 12 | 36 | 191 |
| [pressure-feature-four-transport-classes.csv](pressure-feature-four-transport-classes.csv) | 12 | 48 | 200 |
| [magnetometer-feature.csv](magnetometer-feature.csv) | 12 | 36 | 191 |
| [calibration-fraction.csv](calibration-fraction.csv) | 3 | 9 | 205 |

Files are UTF-8, comma-delimited, with a single header row and quoted fields
where needed. Scores are fractions in [0, 1], exported without deliberate
rounding. Blank cells mean unavailable or inapplicable, not zero. Boolean
fields use `true` and `false`. JSON-valued cells preserve lists and structured
confusion rows. Column sets differ because historical runs recorded different
parameters and selected different model families.

## Source selection

- **Initial sensor/window sweep:** local database
  `D:/tmd-data/all-tmd-work/mlflow.db`, experiment `ALL-TMD Transfer v2` (ID 2),
  and `D:/tmd-data/all-tmd-work/mlartifacts/2/<run_id>/artifacts/`.
  Select active, finished runs with a recorded holdout macro F1. This excludes
  the `artifact-storage-check` run. The 12 remaining runs cover all three
  sensor sets and four window sizes. The local database and artifacts are
  available on this machine but are not archived in this Git repository.
- **Pressure/four-class sweep:**
  `aws-results/20260809T055942Z-9c676a2c-full/`.
- **Magnetometer sweep:**
  `aws-results/20260812T125950Z-a46cc6a0-full/`.
- **Calibration-fraction sweep:**
  `aws-results/20260814T114054Z-c11f9a1a-full/`.

For each AWS bundle, select active, finished runs in its `ALL-TMD` experiment
whose start timestamps fall within `run/run-summary.json`'s execution interval.
Later downloaded databases contain earlier runs too; those are excluded.
Read metrics and splits from `mlflow/mlartifacts/<experiment_id>/<run_id>/artifacts/`,
not the shared `work` reports. In particular, all three calibration trials
share one feature hash, and the work-directory metrics and split were
overwritten by the last trial.

`run_id` is the full MLflow run ID; `aws_run_id` identifies the enclosing AWS
sweep. `experiment_name` is the descriptive experiment requested for this
export; `mlflow_experiment_name` preserves MLflow's actual name. `trial_index`
is the recorded zero-based index, not a new row number. Rows are sorted by
trial index, run ID, and class label ID.

## Metric definitions

| Column or prefix | Meaning |
| --- | --- |
| `support` | True holdout window count for the transport mode. |
| `precision`, `recall`, `f1` | Original classification-report values for this mode. |
| `accuracy` | Overall multiclass holdout accuracy, repeated for each mode in the run. `accuracy_scope` makes this explicit. |
| `one_vs_rest_accuracy` | Derived `(TP + TN) / N`, treating this mode as positive and all others as negative. |
| `specificity` | Derived `TN / (TN + FP)`. |
| `true_positive`, `false_positive`, `false_negative`, `true_negative` | One-vs-rest counts derived from the recorded confusion matrix. |
| `predicted_support`, `support_fraction` | Predicted count for the mode, and true support divided by holdout window count. |
| `precision_defined`, `recall_defined`, `f1_defined` | Whether the corresponding denominator is nonzero. Recorded zero-division scores remain zero. |
| `class_status` | `supported` or `zero_true_support`. |
| `confusion_row_json` | Counts by predicted mode for this true mode. |
| `calibration_*` | Corresponding per-mode metrics on collector calibration windows. `calibration_accuracy` is overall calibration accuracy. |
| `collector_holdout.*`, `collector_calibration.*` | Recorded overall scores, row/group counts, macro/weighted averages, and phone-position breakdowns for that partition. |
| `cross_validation.*`, `best_cross_validation_macro_f1` | Recorded pooled grouped out-of-fold model-selection results. Per-mode CV metrics were not saved. |

Support is **windows**, not sessions, participants, or independent observations.
Phone-position breakdowns are overall metrics for that position, repeated on
each mode row; the artifacts do not provide position-by-mode classification
reports. Do not sum repeated overall counts or average repeated scores as if
they were independent observations.

## Configuration and provenance

Sensor lists, window/step durations, feature names, model family, selected
hyperparameters (`best_params.*`), recorded MLflow parameters
(`mlflow_param.*`), Optuna trial counts by state/family, the best Optuna trial,
run timing, source/collector row counts, split metadata, and available
collector-session fingerprints are included.

`config.*` contains the non-training part of the saved work `trial.json`.
Its canonical SHA-256 was checked against each run's `config_hash`, and its
feature list was checked against the run-specific metrics. These historical
hashes exclude `training`; a matching hash does not authenticate the saved
training settings. `work_snapshot.training.*` therefore explicitly preserves
the shared work snapshot rather than claiming it is an immutable executed
training configuration. In the calibration sweep that snapshot contains
condition C's fractions even on the A/B rows. Use the run-specific
`calibration_fraction.*`, `mode_calibration_fraction`, and
`mode_calibration_groups`/`mode_holdout_groups` fields for those trials; these
come from their separate split artifacts and are cross-checked against MLflow.
`mode_realized_calibration_group_fraction` accounts for whole-group rounding.
The scalar 0.5 setting for older sweeps is retained only in the work-snapshot
columns because it was not recorded in their run-specific split manifests.

`source_*` columns locate the input metrics, splits, trial snapshot, MLflow
database, Optuna table, and AWS summary. Paths under this repository are
relative to `ml/all-tmd-v1`; local external paths are absolute. SHA-256 columns
fingerprint metrics, splits, and trial snapshots. AWS execution commits come
from the run summaries. The local execution commit is unavailable and blank.

## Historical details and validation

- All eight pressure-enabled four-class runs have zero true tram support.
  They still predict tram, and the saved macro F1 averages all four configured
  classes, including tram's zero F1. Balanced accuracy averages recall only
  over classes with true support. The exports preserve both definitions.
- Local winners comprise 6 MLP, 3 random forest, and 3 XGBoost models. Pressure
  winners comprise 2 random forest and 10 XGBoost models. All magnetometer and
  calibration winners are XGBoost. These artifact results supersede the
  blanket XGBoost statement in the preliminary narrative.
- Pressure changes eligible cohorts, calibration fractions change holdout
  membership, and the initial experiment uses US-TMD while the others use
  NOR-TMD. These are not evaluations on one common test set.
- Every per-mode precision, recall, F1, support, overall accuracy, balanced
  accuracy, macro/weighted average, and derived confusion count was checked
  against the original confusion matrix. Split row counts and disjointness,
  selected Optuna scores/families, run counts, source hashes, and exported
  cell values were also checked. There are 39 unique runs and 129 unique
  `(run_id, transport_mode)` rows.
