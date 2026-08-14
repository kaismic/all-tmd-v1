from __future__ import annotations

import pandas as pd
import pytest

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
    assert calibration | holdout == set(
        frame.index[frame["domain"] == "collector"]
    )
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


def test_calibration_fraction_is_applied_per_transport_mode(config_factory):
    fractions = {"bus": 0.8, "car": 0.4, "train": 0.4}
    config = config_factory(calibration_fraction=fractions)
    rows = [
        _row("us-tmd", f"source-{index}", index % 3)
        for index in range(9)
    ]
    for label in range(3):
        for group_number in range(5):
            rows.append(
                _row("collector", f"collector-{label}-{group_number}", label)
            )
    frame = pd.DataFrame(rows).reset_index(drop=True)

    first_manifest = create_splits(frame, config)
    second_manifest = create_splits(frame, config)

    assert first_manifest == second_manifest
    assert first_manifest["manifest_version"] == 3
    assert first_manifest["calibration_fraction_by_label"] == fractions
    assert first_manifest["collector_calibration_group_counts_by_label"] == {
        "bus": 4,
        "car": 2,
        "train": 2,
    }
    assert first_manifest["collector_holdout_group_counts_by_label"] == {
        "bus": 1,
        "car": 3,
        "train": 3,
    }


def test_per_mode_group_counts_are_clamped_for_two_group_modes(config_factory):
    config = config_factory(
        calibration_fraction={"bus": 0.8, "car": 0.4, "train": 0.4}
    )
    rows = [
        _row("us-tmd", f"source-{index}", index % 3)
        for index in range(9)
    ]
    for label in range(3):
        for group_number in range(2):
            rows.append(_row("collector", f"collector-{label}-{group_number}", label))
    frame = pd.DataFrame(rows).reset_index(drop=True)

    manifest = create_splits(frame, config)

    assert manifest["collector_calibration_group_counts_by_label"] == {
        "bus": 1,
        "car": 1,
        "train": 1,
    }
    assert manifest["collector_holdout_group_counts_by_label"] == {
        "bus": 1,
        "car": 1,
        "train": 1,
    }


def test_every_configured_mode_requires_two_collector_groups(config_factory):
    config = config_factory()
    rows = [
        _row("us-tmd", f"source-{index}", index % 3)
        for index in range(9)
    ]
    for label, group_count in {0: 1, 1: 2, 2: 2}.items():
        for group_number in range(group_count):
            rows.append(
                _row("collector", f"collector-{label}-{group_number}", label)
            )
    frame = pd.DataFrame(rows).reset_index(drop=True)

    with pytest.raises(ValueError, match="insufficient: bus"):
        create_splits(frame, config)


def _row(domain: str, group_id: str, label: int) -> dict:
    return {
        "domain": domain,
        "group_id": group_id,
        "label": label,
        "session_id": group_id,
        "window_start_ms": 0,
        "window_end_ms": 1000,
    }
