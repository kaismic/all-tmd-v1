from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
