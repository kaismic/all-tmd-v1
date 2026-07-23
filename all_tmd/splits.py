from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from all_tmd.config import PipelineConfig


def create_splits(frame: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    source_mask = frame["domain"].astype(str) == config.trial.train_dataset
    collector_mask = frame["domain"].astype(str) == "collector"
    source_indices = frame.index[source_mask].astype(int).tolist()
    collector = frame.loc[collector_mask].copy()
    if not source_indices:
        raise ValueError(
            f"No {config.trial.train_dataset} feature rows are available for training"
        )
    if collector.empty:
        raise ValueError("Collector feature rows are required for calibration and testing")

    group_labels = (
        collector.groupby("group_id", sort=True)["label"]
        .agg(lambda values: sorted(set(int(value) for value in values)))
    )
    mixed = group_labels[group_labels.map(len) != 1]
    if not mixed.empty:
        raise ValueError("Each collector group must contain exactly one label")
    groups = group_labels.index.to_numpy(dtype=str)
    labels = np.array([values[0] for values in group_labels], dtype=np.int64)
    class_counts = pd.Series(labels).value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Collector calibration/holdout split requires at least two groups per class"
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=config.trial.training.calibration_fraction,
        random_state=config.trial.training.random_seed,
    )
    try:
        calibration_group_positions, holdout_group_positions = next(
            splitter.split(groups, labels)
        )
    except ValueError as exc:
        raise ValueError(
            "Collector calibration_fraction does not leave at least one group "
            "per class in both calibration and holdout"
        ) from exc
    calibration_groups = set(groups[calibration_group_positions])
    holdout_groups = set(groups[holdout_group_positions])
    calibration_indices = collector.index[
        collector["group_id"].astype(str).isin(calibration_groups)
    ].astype(int).tolist()
    holdout_indices = collector.index[
        collector["group_id"].astype(str).isin(holdout_groups)
    ].astype(int).tolist()

    calibration = frame.loc[calibration_indices]
    calibration_counts = (
        calibration.drop_duplicates("group_id")["label"].value_counts()
    )
    folds = min(5, int(calibration_counts.min()))
    if folds < 2:
        raise ValueError(
            "Collector calibration set requires at least two groups per class "
            "for grouped cross-validation"
        )
    cv = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.trial.training.random_seed,
    )
    cv_folds: list[dict[str, list[int]]] = []
    calibration_array = np.array(calibration_indices, dtype=np.int64)
    for train_positions, valid_positions in cv.split(
        calibration,
        calibration["label"].to_numpy(dtype=np.int64),
        calibration["group_id"].astype(str).to_numpy(),
    ):
        cv_folds.append(
            {
                "train_indices": calibration_array[train_positions].astype(int).tolist(),
                "valid_indices": calibration_array[valid_positions].astype(int).tolist(),
            }
        )

    return {
        "manifest_version": 1,
        "frame_fingerprint": frame_fingerprint(frame),
        "source_indices": source_indices,
        "collector_calibration_indices": calibration_indices,
        "collector_holdout_indices": holdout_indices,
        "collector_cv_folds": cv_folds,
        "group_column": "group_id",
    }


def write_splits(
    manifest: dict[str, Any],
    config: PipelineConfig,
) -> Path:
    path = config.run_dir() / "splits" / f"{config.trial.train_dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def frame_fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        "domain",
        "group_id",
        "label",
        "session_id",
        "window_start_ms",
        "window_end_ms",
    ]
    hashed = pd.util.hash_pandas_object(frame.loc[:, columns], index=False)
    return hashlib.sha256(hashed.to_numpy(dtype=np.uint64).tobytes()).hexdigest()
