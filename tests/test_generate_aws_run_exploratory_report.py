from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import json


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "generate-aws-run-exploratory-report.py"
)
SPEC = importlib.util.spec_from_file_location("generate_aws_run_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def trial(index: int, **params: str):
    metrics = {
        "best_cross_validation_macro_f1": 0.8,
        "collector_holdout": {
            "macro_f1": 0.75,
            "balanced_accuracy": 0.76,
            "accuracy": 0.77,
            "classification_report": {
                "bus": {"f1-score": 0.7},
                "car": {"f1-score": 0.8},
                "accuracy": 0.77,
                "macro avg": {"f1-score": 0.75},
                "weighted avg": {"f1-score": 0.76},
            },
        },
    }
    return MODULE.TrialResult(
        run_uuid=f"run-{index}",
        trial_index=index,
        params={"window_seconds": "60", "step_seconds": "30", **params},
        metrics=metrics,
    )


def test_report_kind_detects_each_supported_sweep() -> None:
    pressure = [
        trial(0),
        trial(1, **{"features.pressure": "standard_deviation,range"}),
    ]
    magnetometer = [
        trial(0, **{"features.magnetometer": "standard_deviation,range"}),
        trial(
            1,
            **{
                "features.magnetometer": "standard_deviation,range,minimum",
            },
        ),
    ]
    calibration = [
        trial(
            0,
            **{
                "calibration_fraction.bus": "0.8",
                "calibration_fraction.car": "0.4",
            },
        )
    ]

    assert MODULE.report_kind(pressure) == "pressure"
    assert MODULE.report_kind(magnetometer) == "magnetometer"
    assert MODULE.report_kind(calibration) == "calibration"


def test_variant_labels_are_descriptive_and_chart_labels_are_compact() -> None:
    pressure_trial = trial(
        0,
        **{
            "features.pressure": (
                "standard_deviation,range,delta_from_session_baseline"
            )
        },
    )
    magnetometer_trial = trial(
        1,
        **{
            "features.magnetometer": "standard_deviation,range,minimum,mean",
        },
    )

    pressure_label = MODULE.variant_label(pressure_trial, "pressure")
    magnetometer_label = MODULE.variant_label(magnetometer_trial, "magnetometer")

    assert pressure_label == "Std. dev. + range + session-baseline delta"
    assert MODULE.chart_variant_label(pressure_label, "pressure") == "Session-baseline delta"
    assert magnetometer_label == "Std. dev. + range + minimum + mean"
    assert MODULE.chart_variant_label(magnetometer_label, "magnetometer") == "+ minimum + mean"


def test_parameter_summary_describes_configuration_dependent_counts() -> None:
    trials = [
        trial(0, collector_session_count="140"),
        trial(1, collector_session_count="104"),
    ]

    assert (
        MODULE._parameter_summary(trials, "collector_session_count")
        == "104--140 (configuration-dependent)"
    )


def test_legacy_snapshot_reconstructs_exact_logged_membership(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "downloaded_sessions"
    relative_paths = [
        Path("raw/participant_001/device-a/session-a.json.gz"),
        Path("raw/participant_002/device-b/session-b.json.gz"),
    ]
    metadata = [
        {
            "session_id": "session-a",
            "participant_id": "participant_001",
            "vehicle_type": "bus",
            "duration_seconds": 120,
            "sample_count": 1000,
        },
        {
            "session_id": "session-b",
            "participant_id": "participant_002",
            "vehicle_type": "car",
            "duration_seconds": 180,
            "sample_count": 2000,
        },
    ]
    for relative_path, payload in zip(relative_paths, metadata, strict=True):
        metadata_path = sessions_dir / relative_path
        metadata_path = metadata_path.with_suffix(
            f"{metadata_path.suffix}.metadata.json"
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            f"ingested path=/data/downloaded_sessions/{path.as_posix()}"
            for path in relative_paths
        ),
        encoding="utf-8",
    )

    snapshot = MODULE._load_legacy_snapshot(log_path, sessions_dir)
    rows = MODULE.build_snapshot_rows(snapshot)

    assert len(snapshot.sessions) == 2
    assert len(snapshot.session_id_digest) == 64
    assert [(row.mode, row.sessions) for row in rows] == [
        ("Bus", 1),
        ("Car", 1),
        ("Total", 2),
    ]
    assert rows[-1].duration == "00:05:00"
    assert rows[-1].samples == 3000


def test_snapshot_failure_renders_explanation() -> None:
    rendered = MODULE.render_snapshot_latex(
        "20260812T125950Z-a46cc6a0-full",
        [trial(0, collector_session_count="152")],
        None,
        "metadata sidecars are unavailable",
    )

    assert "Collector snapshot unavailable" in rendered
    assert "metadata sidecars are unavailable" in rendered


def test_effective_snapshot_matches_mlflow_count_and_digest() -> None:
    sessions = (
        {
            "session_id": "bus-kept",
            "vehicle_type": "bus",
            "participant_id": "participant_001",
            "duration_seconds": 60,
            "sample_count": 100,
        },
        {
            "session_id": "bus-filtered",
            "vehicle_type": "bus",
            "participant_id": "participant_002",
            "duration_seconds": 30,
            "sample_count": 50,
        },
        {
            "session_id": "car-kept",
            "vehicle_type": "car",
            "participant_id": "participant_002",
            "duration_seconds": 120,
            "sample_count": 200,
        },
        {
            "session_id": "tram-not-labelled",
            "vehicle_type": "tram",
            "participant_id": "participant_003",
            "duration_seconds": 180,
            "sample_count": 300,
        },
    )
    expected_sessions = (sessions[0], sessions[2])
    expected_digest = MODULE._session_id_digest(expected_sessions)
    raw_snapshot = MODULE.CollectorSnapshot(sessions, "test snapshot", "raw-digest")
    result = MODULE.derive_effective_snapshot(
        raw_snapshot,
        [
            trial(
                0,
                collector_session_count="2",
                collector_session_digest=expected_digest,
            )
        ],
    )

    assert [session["session_id"] for session in result.sessions] == [
        "bus-kept",
        "car-kept",
    ]
    assert result.session_id_digest == expected_digest
    assert MODULE.build_snapshot_rows(result)[-1].samples == 300
