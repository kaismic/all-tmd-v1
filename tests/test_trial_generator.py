from __future__ import annotations

import json

import pytest

from all_tmd.trial_generator import (
    TrialParametersError,
    generate_trials,
    write_trials,
)


def _document():
    return {
        "default": {
            "train_dataset": "us-tmd",
            "features": {
                "default_window_seconds": 10,
                "default_step_seconds": 5,
                "sensors": {
                    "accelerometer": ["mean"],
                    "gyroscope": ["range"],
                    "pressure": ["standard_deviation"],
                },
            },
            "training": {"random_seed": 42},
        },
        "dimensions": [
            {
                "name": "sensor-set",
                "options": [
                    {
                        "pick": {
                            "features.sensors": [
                                "accelerometer",
                                "gyroscope",
                            ]
                        }
                    },
                    {
                        "pick": {
                            "features.sensors": [
                                "accelerometer",
                                "gyroscope",
                                "pressure",
                            ]
                        }
                    },
                ],
            },
            {
                "name": "window-and-step",
                "options": [
                    {
                        "set": {
                            "features.default_window_seconds": 10,
                            "features.default_step_seconds": 5,
                        }
                    },
                    {
                        "set": {
                            "features.default_window_seconds": 20,
                            "features.default_step_seconds": 10,
                        }
                    },
                ],
            },
        ],
    }


def test_generates_cartesian_product_while_preserving_paired_values():
    document = _document()

    trials = generate_trials(document)

    assert len(trials) == 4
    assert [
        (
            list(trial["features"]["sensors"]),
            trial["features"]["default_window_seconds"],
            trial["features"]["default_step_seconds"],
        )
        for trial in trials
    ] == [
        (["accelerometer", "gyroscope"], 10, 5),
        (["accelerometer", "gyroscope"], 20, 10),
        (["accelerometer", "gyroscope", "pressure"], 10, 5),
        (["accelerometer", "gyroscope", "pressure"], 20, 10),
    ]
    assert "pressure" in document["default"]["features"]["sensors"]


def test_empty_dimensions_generate_one_copy_of_default():
    document = {"default": {"training": {"random_seed": 42}}, "dimensions": []}

    trials = generate_trials(document)

    assert trials == [document["default"]]
    assert trials[0] is not document["default"]


def test_write_trials_outputs_formatted_json(tmp_path):
    output = tmp_path / "nested" / "trials.json"
    trials = generate_trials(_document())

    write_trials(trials, output)

    assert json.loads(output.read_text(encoding="utf-8")) == trials
    assert output.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document["dimensions"][0]["options"][0]["pick"].update(
                {"features.unknown": ["anything"]}
            ),
            "does not exist",
        ),
        (
            lambda document: document["dimensions"][0]["options"][0]["pick"][
                "features.sensors"
            ].append("unknown"),
            "unknown key",
        ),
        (
            lambda document: document["dimensions"][0]["options"][1].update(
                {"set": {"training.random_seed": 123}}
            ),
            "same operations and paths",
        ),
        (
            lambda document: document["dimensions"].append(
                {
                    "name": "duplicate-window",
                    "options": [
                        {
                            "set": {
                                "features.default_window_seconds": 60,
                            }
                        }
                    ],
                }
            ),
            "overlapping paths",
        ),
    ],
)
def test_rejects_invalid_or_ambiguous_dimensions(mutate, message):
    document = _document()
    mutate(document)

    with pytest.raises(TrialParametersError, match=message):
        generate_trials(document)
