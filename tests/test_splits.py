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


def _row(domain: str, group_id: str, label: int) -> dict:
    return {
        "domain": domain,
        "group_id": group_id,
        "label": label,
        "session_id": group_id,
        "window_start_ms": 0,
        "window_end_ms": 1000,
    }
