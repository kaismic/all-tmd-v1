from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from all_tmd.config import PipelineConfig


def start_run(config: PipelineConfig):
    if not config.mlflow.enabled:
        return _NullRun()
    import mlflow

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    run = mlflow.start_run(
        run_name=f"{config.trial.train_dataset}-{config.config_hash[:8]}"
    )
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
        }
    )
    return run


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


class _NullRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
