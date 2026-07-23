from __future__ import annotations

from pathlib import Path
from typing import Any

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


class _NullRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
