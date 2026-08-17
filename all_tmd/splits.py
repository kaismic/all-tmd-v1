from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from all_tmd.config import PipelineConfig


def create_splits(frame: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    if config.trial.training.evaluation_strategy == "participant_nested_cv":
        return _create_participant_nested_splits(frame, config)
    return _create_session_holdout_splits(frame, config)


def _create_session_holdout_splits(
    frame: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
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
    group_classes = group_labels.map(lambda values: values[0])
    configured_values = set(config.trial.labels.values())
    unknown_values = sorted(set(group_classes) - configured_values)
    if unknown_values:
        raise ValueError(
            "Collector contains label value(s) not configured by the trial: "
            + ", ".join(str(value) for value in unknown_values)
        )

    calibration_groups: set[str] = set()
    holdout_groups: set[str] = set()
    calibration_group_counts: dict[str, int] = {}
    holdout_group_counts: dict[str, int] = {}
    rng = np.random.default_rng(config.trial.training.random_seed)
    ordered_labels = sorted(
        config.trial.labels.items(),
        key=lambda item: (item[1], item[0]),
    )
    sparse_modes: list[str] = []
    for mode, label_value in ordered_labels:
        mode_groups = group_classes.index[
            group_classes == label_value
        ].to_numpy(dtype=str)
        if len(mode_groups) < 2:
            sparse_modes.append(mode)
            continue
        shuffled_groups = rng.permutation(mode_groups)
        requested_fraction = config.trial.training.calibration_fraction[mode]
        calibration_count = max(
            1,
            min(
                len(mode_groups) - 1,
                math.floor(len(mode_groups) * requested_fraction),
            ),
        )
        calibration_groups.update(shuffled_groups[:calibration_count].tolist())
        holdout_groups.update(shuffled_groups[calibration_count:].tolist())
        calibration_group_counts[mode] = calibration_count
        holdout_group_counts[mode] = len(mode_groups) - calibration_count
    if sparse_modes:
        raise ValueError(
            "Collector calibration/holdout split requires at least two groups "
            "for each configured transport mode; insufficient: "
            + ", ".join(sparse_modes)
        )
    calibration_indices = collector.index[
        collector["group_id"].astype(str).isin(calibration_groups)
    ].astype(int).tolist()
    holdout_indices = collector.index[
        collector["group_id"].astype(str).isin(holdout_groups)
    ].astype(int).tolist()

    calibration = frame.loc[calibration_indices]
    calibration_group_count = int(calibration["group_id"].nunique())
    folds = min(5, calibration_group_count)
    if folds < 2:
        raise ValueError(
            "Collector calibration set requires at least two groups for "
            "grouped cross-validation"
        )
    cv = GroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.trial.training.random_seed,
    )
    cv_folds: list[dict[str, list[int]]] = []
    calibration_array = np.array(calibration_indices, dtype=np.int64)
    for train_positions, valid_positions in cv.split(
        calibration,
        groups=calibration["group_id"].astype(str).to_numpy(),
    ):
        cv_folds.append(
            {
                "train_indices": calibration_array[train_positions].astype(int).tolist(),
                "valid_indices": calibration_array[valid_positions].astype(int).tolist(),
            }
        )

    return {
        "manifest_version": 3,
        "frame_fingerprint": frame_fingerprint(frame),
        "source_indices": source_indices,
        "collector_calibration_indices": calibration_indices,
        "collector_holdout_indices": holdout_indices,
        "collector_cv_folds": cv_folds,
        "group_column": "group_id",
        "calibration_fraction_by_label": dict(
            config.trial.training.calibration_fraction
        ),
        "collector_calibration_group_counts_by_label": calibration_group_counts,
        "collector_holdout_group_counts_by_label": holdout_group_counts,
        "evaluation_strategy": "session_holdout",
    }


def _create_participant_nested_splits(
    frame: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    source_mask = frame["domain"].astype(str) == config.trial.train_dataset
    collector_mask = frame["domain"].astype(str) == "collector"
    source_indices = frame.index[source_mask].astype(int).tolist()
    collector = frame.loc[collector_mask].copy()
    if not source_indices:
        raise ValueError(
            f"No {config.trial.train_dataset} feature rows are available for training"
        )
    if collector.empty:
        raise ValueError("Collector feature rows are required for evaluation")
    if collector["participant_id"].isna().any():
        raise ValueError("Collector participant IDs are required for participant CV")

    configured_values = set(config.trial.labels.values())
    unknown_values = sorted(set(collector["label"].astype(int)) - configured_values)
    if unknown_values:
        raise ValueError(
            "Collector contains label value(s) not configured by the trial: "
            + ", ".join(str(value) for value in unknown_values)
        )
    label_participants = collector.groupby("label")["participant_id"].nunique()
    value_to_name = {value: name for name, value in config.trial.labels.items()}
    sparse_modes = [
        value_to_name[label]
        for label in sorted(configured_values)
        if int(label_participants.get(label, 0)) < 2
    ]
    if sparse_modes:
        raise ValueError(
            "Participant-independent evaluation requires at least two collector "
            "participants for every mode; insufficient: " + ", ".join(sparse_modes)
        )

    participants = sorted(collector["participant_id"].astype(str).unique())
    if len(participants) < 3:
        raise ValueError(
            "Participant-independent nested evaluation requires at least three "
            "collector participants"
        )
    outer_folds: list[dict[str, Any]] = []
    for outer_number, held_out_participant in enumerate(participants):
        test_mask = collector["participant_id"].astype(str) == held_out_participant
        train_indices = collector.index[~test_mask].astype(int).tolist()
        test_indices = collector.index[test_mask].astype(int).tolist()
        inner_folds = _participant_folds(
            frame.loc[train_indices],
            config,
            random_seed=config.trial.training.random_seed + outer_number,
        )
        outer_folds.append(
            {
                "held_out_participant_id": held_out_participant,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "inner_folds": inner_folds,
            }
        )

    collector_indices = collector.index.astype(int).tolist()
    return {
        "manifest_version": 4,
        "evaluation_strategy": "participant_nested_cv",
        "frame_fingerprint": frame_fingerprint(frame),
        "source_indices": source_indices,
        "collector_evaluation_indices": collector_indices,
        "participant_outer_folds": outer_folds,
        "final_participant_cv_folds": _participant_folds(
            collector,
            config,
            random_seed=config.trial.training.random_seed,
        ),
        "group_column": "participant_id",
        "collector_participant_count": len(participants),
        "collector_participants_by_label": {
            value_to_name[label]: int(label_participants.get(label, 0))
            for label in sorted(configured_values)
        },
    }


def _participant_folds(
    collector: pd.DataFrame,
    config: PipelineConfig,
    *,
    random_seed: int,
) -> list[dict[str, list[int]]]:
    participant_count = int(collector["participant_id"].nunique())
    folds = min(config.trial.training.participant_inner_folds, participant_count)
    if folds < 2:
        raise ValueError("Participant cross-validation requires at least two groups")
    cv = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_seed,
    )
    index_array = collector.index.to_numpy(dtype=np.int64)
    result: list[dict[str, list[int]]] = []
    for train_positions, valid_positions in cv.split(
        collector,
        y=collector["label"].to_numpy(dtype=np.int64),
        groups=collector["participant_id"].astype(str).to_numpy(),
    ):
        result.append(
            {
                "train_indices": index_array[train_positions].astype(int).tolist(),
                "valid_indices": index_array[valid_positions].astype(int).tolist(),
            }
        )
    return result


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
        "participant_id",
        "group_id",
        "label",
        "session_id",
        "window_start_ms",
        "window_end_ms",
    ]
    hashed = pd.util.hash_pandas_object(
        frame.reindex(columns=columns, fill_value=""),
        index=False,
    )
    return hashlib.sha256(hashed.to_numpy(dtype=np.uint64).tobytes()).hexdigest()
