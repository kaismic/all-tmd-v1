from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from all_tmd.mlflow_utils import (
    collector_session_summary,
    dataset_digest,
    log_confusion_matrix,
    mlflow_run_name,
    start_run,
)


FEATURE_NAMES = ["accelerometer#mean"]


def test_mlflow_run_name_includes_utc_start_time_and_collector_count(
    config_factory,
):
    config = config_factory()

    name = mlflow_run_name(
        config,
        42,
        datetime(2026, 8, 12, 3, 45, 12, tzinfo=timezone.utc),
    )

    assert name == f"us-tmd-{config.config_hash[:8]}-20260812T034512Z-42"


def test_collector_session_summary_is_order_independent():
    frame = _dataset_frame()

    first = collector_session_summary(frame)
    second = collector_session_summary(frame.sample(frac=1, random_state=7))

    assert first == second
    assert first[1] == 2


def test_collector_session_summary_changes_with_session_membership():
    frame = _dataset_frame()
    original_digest, original_count = collector_session_summary(frame)
    added = pd.concat(
        [
            frame,
            pd.DataFrame(
                [_row("collector", "collector-3", 1, 3000, 4.0)]
            ),
        ],
        ignore_index=True,
    )

    added_digest, added_count = collector_session_summary(added)
    removed_digest, removed_count = collector_session_summary(frame.iloc[:-1])

    assert added_digest != original_digest
    assert added_count == original_count + 1
    assert removed_digest != original_digest
    assert removed_count == original_count - 1


def test_dataset_digest_is_order_independent():
    frame = _dataset_frame()

    digest = dataset_digest(frame, FEATURE_NAMES)

    assert digest == dataset_digest(
        frame.sample(frac=1, random_state=11),
        FEATURE_NAMES,
    )
    assert len(digest) == 36


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("accelerometer#mean", 99.0),
        ("label", 2),
        ("group_id", "changed-group"),
        ("window_start_ms", 999),
        ("window_end_ms", 1999),
    ],
)
def test_dataset_digest_changes_with_content(column, value):
    frame = _dataset_frame()
    changed = frame.copy()
    changed.loc[0, column] = value

    assert dataset_digest(changed, FEATURE_NAMES) != dataset_digest(
        frame,
        FEATURE_NAMES,
    )


def test_start_run_logs_dataset_inputs_and_collector_summary(
    config_factory,
    monkeypatch,
):
    config = config_factory(
        mlflow_enabled=True,
        sensors={
            "accelerometer": ["mean", "standard deviation"],
            "pressure": ["range"],
        },
        collector_minimum_sampling_rate={
            "accelerometer": 30,
            "pressure": 2,
        },
        collector_max_sample_interval_ms=500,
    )
    frame = _dataset_frame()
    manifest = {
        "source_indices": [0, 1],
        "collector_calibration_indices": [2],
        "collector_holdout_indices": [3],
    }
    recorded = {
        "inputs": [],
        "params": {},
    }

    def from_pandas(dataset_frame, **kwargs):
        return SimpleNamespace(frame=dataset_frame.copy(), **kwargs)

    def fake_start_run(**kwargs):
        recorded["run_name"] = kwargs["run_name"]
        return nullcontext("active-run")

    fake_mlflow = SimpleNamespace(
        data=SimpleNamespace(from_pandas=from_pandas),
        set_tracking_uri=lambda uri: recorded.setdefault("tracking_uri", uri),
        set_experiment=lambda name: recorded.setdefault("experiment", name),
        start_run=fake_start_run,
        log_params=lambda params: recorded["params"].update(params),
        log_input=lambda dataset, context: recorded["inputs"].append(
            (dataset, context)
        ),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with start_run(config, frame, manifest) as run:
        assert run == "active-run"

    assert recorded["experiment"] == "test"
    assert recorded["run_name"].startswith(
        f"us-tmd-{config.config_hash[:8]}-"
    )
    assert recorded["run_name"].endswith("Z-2")
    assert recorded["params"]["sensors"] == "accelerometer,pressure"
    assert recorded["params"]["features.accelerometer"] == (
        "mean,standard_deviation"
    )
    assert recorded["params"]["features.pressure"] == "range"
    assert "features.gyroscope" not in recorded["params"]
    assert recorded["params"]["feature_names"] == (
        "accelerometer#mean,accelerometer#standard_deviation,pressure#range"
    )
    assert recorded["params"]["collector_session_count"] == 2
    assert len(recorded["params"]["collector_session_digest"]) == 64
    assert recorded["params"]["collector_max_sample_interval_ms"] == 500
    assert (
        recorded["params"]["collector_minimum_sampling_rate.accelerometer"]
        == 30
    )
    assert recorded["params"]["collector_minimum_sampling_rate.pressure"] == 2
    assert [context for _, context in recorded["inputs"]] == [
        "training",
        "calibration",
        "evaluation",
    ]
    assert [dataset.name for dataset, _ in recorded["inputs"]] == [
        "us-tmd-training-features",
        "collector-calibration-features",
        "collector-holdout-features",
    ]
    assert [
        dataset.frame["session_id"].tolist()
        for dataset, _ in recorded["inputs"]
    ] == [["source-1", "source-2"], ["collector-1"], ["collector-2"]]
    assert all(len(dataset.digest) == 36 for dataset, _ in recorded["inputs"])


def test_start_run_disabled_does_not_import_mlflow(config_factory, monkeypatch):
    config = config_factory(mlflow_enabled=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with start_run(config, _dataset_frame(), {}) as run:
        assert run is None


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("us-tmd", "source-1", 0, 0, 1.0),
            _row("us-tmd", "source-2", 1, 1000, 2.0),
            _row("collector", "collector-1", 0, 1000, 2.5),
            _row("collector", "collector-2", 1, 2000, 3.0),
        ]
    )


def _row(
    domain: str,
    session_id: str,
    label: int,
    window_start_ms: int,
    value: float,
) -> dict:
    return {
        "domain": domain,
        "group_id": session_id,
        "label": label,
        "session_id": session_id,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_start_ms + 1000,
        "accelerometer#mean": value,
        "accelerometer#standard_deviation": value / 10,
        "pressure#range": value * 10,
    }


@pytest.mark.parametrize(
    ("normalize", "expected", "expected_format"),
    [
        (
            False,
            np.array([[3, 1], [2, 4]]),
            "3",
        ),
        (
            True,
            np.array([[0.75, 0.25], [1 / 3, 2 / 3]]),
            "0.75",
        ),
    ],
)
def test_log_confusion_matrix_logs_figure(
    monkeypatch,
    normalize,
    expected,
    expected_format,
):
    logged = {}

    def capture_figure(figure, artifact_file):
        logged["artifact_file"] = artifact_file
        logged["values"] = np.asarray(figure.axes[0].images[0].get_array())
        logged["annotations"] = {
            text.get_text() for text in figure.axes[0].texts
        }

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(log_figure=capture_figure),
    )

    log_confusion_matrix(
        [[3, 1], [2, 4]],
        ["bus", "car"],
        "evaluation/confusion-matrix.png",
        normalize=normalize,
    )

    assert logged["artifact_file"] == "evaluation/confusion-matrix.png"
    np.testing.assert_allclose(logged["values"], expected)
    assert expected_format in logged["annotations"]


def test_log_confusion_matrix_rejects_wrong_dimensions(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(log_figure=lambda *_args: None),
    )

    with pytest.raises(ValueError, match="dimensions"):
        log_confusion_matrix(
            [[1, 2]],
            ["bus", "car"],
            "evaluation/confusion-matrix.png",
        )
