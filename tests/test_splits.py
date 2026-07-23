from __future__ import annotations

import pandas as pd

from all_tmd.splits import create_splits


def test_collector_groups_do_not_leak_between_splits(config_factory):
    config = config_factory(calibration_fraction=0.5)
    rows = [
        _row("us-tmd", f"source-{index}", index % 3)
        for index in range(9)
    ]
    for label in range(3):
        for group_number in range(4):
            group = f"collector-{label}-{group_number}"
            rows.extend([_row("collector", group, label) for _ in range(2)])
    frame = pd.DataFrame(rows).reset_index(drop=True)
    manifest = create_splits(frame, config)

    calibration = set(manifest["collector_calibration_indices"])
    holdout = set(manifest["collector_holdout_indices"])
    assert calibration.isdisjoint(holdout)
    calibration_groups = set(frame.loc[list(calibration), "group_id"])
    holdout_groups = set(frame.loc[list(holdout), "group_id"])
    assert calibration_groups.isdisjoint(holdout_groups)

    for fold in manifest["collector_cv_folds"]:
        train_groups = set(frame.loc[fold["train_indices"], "group_id"])
        valid_groups = set(frame.loc[fold["valid_indices"], "group_id"])
        assert train_groups.isdisjoint(valid_groups)


def test_sparse_class_can_use_one_calibration_group(config_factory):
    config = config_factory(calibration_fraction=0.5)
    rows = [
        _row("us-tmd", f"source-{index}", index % 3)
        for index in range(9)
    ]
    collector_group_counts = {0: 2, 1: 8, 2: 4}
    for label, group_count in collector_group_counts.items():
        for group_number in range(group_count):
            group = f"collector-{label}-{group_number}"
            rows.extend([_row("collector", group, label) for _ in range(2)])
    frame = pd.DataFrame(rows).reset_index(drop=True)

    manifest = create_splits(frame, config)
    calibration = frame.loc[manifest["collector_calibration_indices"]]
    holdout = frame.loc[manifest["collector_holdout_indices"]]

    assert calibration.drop_duplicates("group_id")["label"].value_counts()[0] == 1
    assert holdout.drop_duplicates("group_id")["label"].value_counts()[0] == 1
    validated_indices = [
        index
        for fold in manifest["collector_cv_folds"]
        for index in fold["valid_indices"]
    ]
    assert sorted(validated_indices) == sorted(
        manifest["collector_calibration_indices"]
    )


def _row(domain: str, group_id: str, label: int) -> dict:
    return {
        "domain": domain,
        "group_id": group_id,
        "label": label,
        "session_id": group_id,
        "window_start_ms": 0,
        "window_end_ms": 1000,
    }
