from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from all_tmd.config import PipelineConfig


DATASET_ID_COLUMNS = (
    "domain",
    "group_id",
    "label",
    "session_id",
    "window_start_ms",
    "window_end_ms",
)


@contextmanager
def start_run(
    config: PipelineConfig,
    frame: pd.DataFrame,
    split_manifest: dict[str, Any],
):
    if not config.mlflow.enabled:
        yield None
        return
    import mlflow

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    collector_digest, collector_count = collector_session_summary(frame)
    with mlflow.start_run(
        run_name=f"{config.trial.train_dataset}-{config.config_hash[:8]}"
    ) as run:
        mlflow.log_params(
            {
                "config_hash": config.config_hash,
                "trial_index": config.trial_index,
                "train_dataset": config.trial.train_dataset,
                "window_seconds": config.trial.features.default_window_seconds,
                "step_seconds": config.trial.features.default_step_seconds,
                "sensors": ",".join(config.trial.features.sensors),
                "model_families": ",".join(config.trial.training.model_families),
                "optuna_trials": config.trial.training.optuna_trials,
                "collector_session_digest": collector_digest,
                "collector_session_count": collector_count,
            }
        )
        log_dataset_inputs(config, frame, split_manifest)
        yield run


def collector_session_summary(frame: pd.DataFrame) -> tuple[str, int]:
    collector = frame.loc[
        frame["domain"].astype(str) == "collector",
        "session_id",
    ]
    session_ids = sorted(set(collector.dropna().astype(str)))
    canonical = json.dumps(
        session_ids,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), len(session_ids)


def dataset_digest(frame: pd.DataFrame, feature_names: Sequence[str]) -> str:
    columns = list(DATASET_ID_COLUMNS) + list(feature_names)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(
            "MLflow dataset fingerprint is missing column(s): "
            + ", ".join(missing)
        )
    selected = frame.loc[:, columns]
    row_hashes = pd.util.hash_pandas_object(
        selected,
        index=False,
        categorize=True,
    ).to_numpy(dtype="uint64", copy=True)
    row_hashes.sort()
    header = json.dumps(
        {
            "columns": columns,
            "dtypes": [str(selected[column].dtype) for column in columns],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def log_dataset_inputs(
    config: PipelineConfig,
    frame: pd.DataFrame,
    split_manifest: dict[str, Any],
) -> None:
    import mlflow

    run_dir = config.run_dir()
    feature_names = config.trial.feature_names
    datasets = (
        (
            f"{config.trial.train_dataset}-training-features",
            "training",
            split_manifest["source_indices"],
            run_dir / "features" / config.trial.train_dataset,
        ),
        (
            "collector-calibration-features",
            "calibration",
            split_manifest["collector_calibration_indices"],
            run_dir / "features" / "collector",
        ),
        (
            "collector-holdout-features",
            "evaluation",
            split_manifest["collector_holdout_indices"],
            run_dir / "features" / "collector",
        ),
    )
    columns = list(DATASET_ID_COLUMNS) + feature_names
    for name, context, indices, source in datasets:
        dataset_frame = frame.loc[indices, columns].reset_index(drop=True)
        dataset = mlflow.data.from_pandas(
            dataset_frame,
            source=str(source),
            name=name,
            digest=dataset_digest(dataset_frame, feature_names),
        )
        mlflow.log_input(dataset, context=context)


def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(f"{prefix}{key}", float(value))


def log_artifact(path: Path) -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        return
    if path.exists():
        mlflow.log_artifact(str(path))


def log_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    label_names: Sequence[str],
    artifact_file: str,
    *,
    normalize: bool = False,
) -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        return
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import ConfusionMatrixDisplay

    values = np.asarray(matrix)
    if values.shape != (len(label_names), len(label_names)):
        raise ValueError(
            "Confusion matrix dimensions must match the configured labels"
        )
    if normalize:
        values = values.astype(np.float64)
        row_totals = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_totals,
            out=np.zeros_like(values),
            where=row_totals != 0,
        )

    figure = Figure(figsize=(7, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    try:
        ConfusionMatrixDisplay(
            confusion_matrix=values,
            display_labels=list(label_names),
        ).plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
            values_format=".2f" if normalize else "d",
        )
        title_suffix = " (row normalized)" if normalize else ""
        axis.set_title(f"Collector holdout confusion matrix{title_suffix}")
        figure.tight_layout()
        mlflow.log_figure(figure, artifact_file)
    finally:
        figure.clear()
