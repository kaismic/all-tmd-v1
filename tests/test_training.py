from __future__ import annotations

from contextlib import nullcontext

import pandas as pd

from all_tmd.train import train


def test_training_writes_required_reports(config_factory, monkeypatch):
    config = config_factory(mlflow_enabled=True)
    logged_artifacts = []
    logged_confusion_matrices = []
    monkeypatch.setattr(
        "all_tmd.train.start_run",
        lambda _config: nullcontext(),
    )
    monkeypatch.setattr(
        "all_tmd.train.log_metrics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "all_tmd.train.log_artifact",
        logged_artifacts.append,
    )
    monkeypatch.setattr(
        "all_tmd.train.log_confusion_matrix",
        lambda matrix, labels, artifact_file, **kwargs: (
            logged_confusion_matrices.append(
                (matrix, labels, artifact_file, kwargs)
            )
        ),
    )
    run_dir = config.run_dir()
    source_dir = run_dir / "features" / "us-tmd"
    collector_dir = run_dir / "features" / "collector"
    source_dir.mkdir(parents=True)
    collector_dir.mkdir(parents=True)

    source_rows = []
    for label in range(3):
        for index in range(4):
            source_rows.append(
                _feature_row(
                    "us-tmd",
                    f"source-{label}-{index}",
                    label,
                    label + index / 10,
                )
            )
    collector_rows = []
    for label in range(3):
        for index in range(4):
            collector_rows.append(
                _feature_row(
                    "collector",
                    f"collector-{label}-{index}",
                    label,
                    label + index / 10,
                )
            )
    pd.DataFrame(source_rows).to_parquet(
        source_dir / "part-000000.parquet",
        index=False,
    )
    pd.DataFrame(collector_rows).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )

    metrics = train(config)
    report_dir = run_dir / "reports" / "us-tmd"
    assert metrics["collector_holdout"]["rows"] == 6
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "model.joblib").exists()
    assert (report_dir / "optuna-trials.csv").exists()
    assert {path.name for path in logged_artifacts} == {
        "metrics.json",
        "model.joblib",
        "optuna-trials.csv",
        "us-tmd.json",
    }
    assert [
        (artifact_file, kwargs.get("normalize", False))
        for _, _, artifact_file, kwargs in logged_confusion_matrices
    ] == [
        ("evaluation/collector-holdout-confusion-matrix.png", False),
        (
            "evaluation/collector-holdout-confusion-matrix-normalized.png",
            True,
        ),
    ]
    assert all(
        labels == ["bus", "car", "train"]
        for _, labels, _, _ in logged_confusion_matrices
    )


def _feature_row(
    domain: str,
    group_id: str,
    label: int,
    value: float,
) -> dict:
    return {
        "domain": domain,
        "participant_id": group_id,
        "device_id": "device",
        "session_id": group_id,
        "trip_id": group_id,
        "group_id": group_id,
        "vehicle_type": ("bus", "car", "train")[label],
        "label": label,
        "phone_position": "pocket",
        "window_start_ms": 0,
        "window_end_ms": 1000,
        "accelerometer#mean": value,
    }
