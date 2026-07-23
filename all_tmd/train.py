from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from all_tmd.config import PipelineConfig
from all_tmd.mlflow_utils import log_artifact, log_metrics, start_run
from all_tmd.models import (
    fit_with_optional_sample_weight,
    model_from_params,
    suggest_model_params,
)
from all_tmd.progress import progress
from all_tmd.splits import create_splits, write_splits


def train(config: PipelineConfig) -> dict[str, Any]:
    run_dir = config.run_dir()
    source_name = config.trial.train_dataset
    source_frame = _read_feature_dataset(run_dir / "features" / source_name)
    collector_frame = _read_feature_dataset(run_dir / "features" / "collector")
    frame = pd.concat([source_frame, collector_frame], ignore_index=True)
    feature_names = config.trial.feature_names
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError("Feature dataset is missing columns: " + ", ".join(missing))
    manifest = create_splits(frame, config)
    split_path = write_splits(manifest, config)

    report_dir = run_dir / "reports" / source_name
    report_dir.mkdir(parents=True, exist_ok=True)
    x = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    source_indices = manifest["source_indices"]
    cv_folds = manifest["collector_cv_folds"]
    total_trials = config.trial.training.optuna_trials
    configured_labels = sorted(config.trial.labels.values())

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, config.trial.training.model_families)
        fold_scores: list[float] = []
        out_of_fold_truth: list[np.ndarray] = []
        out_of_fold_predictions: list[np.ndarray] = []
        for fold_number, fold in enumerate(cv_folds):
            train_indices = np.array(
                source_indices + fold["train_indices"],
                dtype=np.int64,
            )
            valid_indices = np.array(fold["valid_indices"], dtype=np.int64)
            model = model_from_params(
                params,
                config.trial.training.random_seed,
                config.training.n_jobs,
                config.training.xgboost_device,
            )
            weights = compute_sample_weight("balanced", y[train_indices])
            fit_with_optional_sample_weight(
                model,
                x[train_indices],
                y[train_indices],
                weights,
            )
            predictions = model.predict(x[valid_indices])
            out_of_fold_truth.append(y[valid_indices])
            out_of_fold_predictions.append(predictions)
            score = float(
                f1_score(
                    y[valid_indices],
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(
            f1_score(
                np.concatenate(out_of_fold_truth),
                np.concatenate(out_of_fold_predictions),
                labels=configured_labels,
                average="macro",
                zero_division=0,
            )
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=config.trial.training.random_seed
        ),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    progress(
        f"Optuna optimization starting: dataset={source_name}, trials={total_trials}"
    )
    with start_run(config):
        study.optimize(
            objective,
            n_trials=total_trials,
            timeout=config.training.timeout_seconds,
            callbacks=[_trial_progress(total_trials)],
        )
        best_params = dict(
            study.best_trial.user_attrs.get("model_params")
            or study.best_trial.params
        )
        final_indices = np.array(
            source_indices + manifest["collector_calibration_indices"],
            dtype=np.int64,
        )
        final_model = model_from_params(
            best_params,
            config.trial.training.random_seed,
            config.training.n_jobs,
            config.training.xgboost_device,
        )
        weights = compute_sample_weight("balanced", y[final_indices])
        calibration_set = set(manifest["collector_calibration_indices"])
        weights = np.array(
            [
                weight * 2.0 if index in calibration_set else weight
                for index, weight in zip(final_indices, weights)
            ],
            dtype=np.float64,
        )
        fit_with_optional_sample_weight(
            final_model,
            x[final_indices],
            y[final_indices],
            weights,
        )

        metrics = {
            "config_hash": config.config_hash,
            "trial_index": config.trial_index,
            "train_dataset": source_name,
            "feature_names": feature_names,
            "best_cross_validation_macro_f1": float(study.best_value),
            "cross_validation": {
                "method": "grouped_out_of_fold",
                "folds": int(len(cv_folds)),
                "rows": int(len(manifest["collector_calibration_indices"])),
                "macro_f1": float(study.best_value),
            },
            "best_params": best_params,
            "collector_calibration": _evaluate(
                final_model,
                x,
                y,
                frame,
                manifest["collector_calibration_indices"],
                config.trial.labels,
            ),
            "collector_holdout": _evaluate(
                final_model,
                x,
                y,
                frame,
                manifest["collector_holdout_indices"],
                config.trial.labels,
            ),
            "rows": int(len(frame)),
            "source_rows": int(len(source_indices)),
            "collector_rows": int(
                len(manifest["collector_calibration_indices"])
                + len(manifest["collector_holdout_indices"])
            ),
            "groups": int(frame["group_id"].nunique()),
        }
        metrics_path = report_dir / "metrics.json"
        trials_path = report_dir / "optuna-trials.csv"
        model_path = report_dir / "model.joblib"
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        study.trials_dataframe().to_csv(trials_path, index=False)
        joblib.dump(final_model, model_path)
        if config.mlflow.enabled:
            log_metrics(
                metrics["collector_holdout"],
                prefix="collector_holdout.",
            )
            log_metrics(
                {"best_cross_validation_macro_f1": study.best_value},
            )
            for artifact in (metrics_path, trials_path, model_path, split_path):
                log_artifact(artifact)
    progress(f"Training complete: dataset={source_name}, reports={report_dir}")
    return metrics


def _read_feature_dataset(path: Path) -> pd.DataFrame:
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No feature Parquet parts found: {path}")
    return pd.concat(
        (pd.read_parquet(part) for part in parts),
        ignore_index=True,
    )


def _trial_progress(total_trials: int):
    def callback(
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        try:
            best = f"{study.best_value:.4f}"
        except ValueError:
            best = "n/a"
        value = "n/a" if trial.value is None else f"{trial.value:.4f}"
        progress(
            f"Optuna trial {trial.number + 1}/{total_trials}: "
            f"state={trial.state.name}, value={value}, best={best}"
        )

    return callback


def _evaluate(
    model,
    x: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    indices: list[int],
    labels: dict[str, int],
) -> dict[str, Any]:
    if not indices:
        return {"rows": 0}
    idx = np.array(indices, dtype=np.int64)
    predictions = model.predict(x[idx])
    ordered_labels = sorted(labels.items(), key=lambda item: item[1])
    label_names = [name for name, _ in ordered_labels]
    label_values = [value for _, value in ordered_labels]
    result: dict[str, Any] = {
        "rows": int(len(idx)),
        "groups": int(frame.loc[idx, "group_id"].nunique()),
        "accuracy": float(accuracy_score(y[idx], predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y[idx], predictions)
        ),
        "macro_f1": float(f1_score(y[idx], predictions, average="macro")),
        "classification_report": classification_report(
            y[idx],
            predictions,
            labels=label_values,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y[idx],
            predictions,
            labels=label_values,
        ).tolist(),
        "by_phone_position": {},
    }
    positions = frame.loc[idx, "phone_position"].astype(str)
    for position in sorted(positions.unique()):
        mask = positions.to_numpy() == position
        result["by_phone_position"][position] = {
            "rows": int(mask.sum()),
            "macro_f1": float(
                f1_score(y[idx][mask], predictions[mask], average="macro")
            ),
        }
    return result
