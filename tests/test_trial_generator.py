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


def test_pick_selects_array_values_in_requested_order_without_mutating_default():
    document = {
        "default": {
            "features": {
                "sensors": {
                    "accelerometer": [
                        "mean",
                        "standard_deviation",
                        "delta_from_window_start",
                    ]
                }
            }
        },
        "dimensions": [
            {
                "name": "accelerometer-features",
                "options": [
                    {
                        "pick": {
                            "features.sensors.accelerometer": [
                                "delta_from_window_start",
                                "mean",
                            ]
                        }
                    }
                ],
            }
        ],
    }

    trials = generate_trials(document)

    assert trials[0]["features"]["sensors"]["accelerometer"] == [
        "delta_from_window_start",
        "mean",
    ]
    assert document["default"]["features"]["sensors"]["accelerometer"] == [
        "mean",
        "standard_deviation",
        "delta_from_window_start",
    ]


def test_nested_sensor_and_feature_picks_deduplicate_excluded_sensor_variants():
    document = {
        "default": {
            "features": {
                "sensors": {
                    "accelerometer": ["mean", "standard_deviation"],
                    "pressure": ["mean", "standard_deviation"],
                }
            }
        },
        "dimensions": [
            {
                "name": "sensor-set",
                "options": [
                    {"pick": {"features.sensors": ["accelerometer"]}},
                    {
                        "pick": {
                            "features.sensors": ["accelerometer", "pressure"]
                        }
                    },
                ],
            },
            {
                "name": "accelerometer-features",
                "options": [
                    {
                        "pick": {
                            "features.sensors.accelerometer": ["mean"]
                        }
                    },
                    {
                        "pick": {
                            "features.sensors.accelerometer": [
                                "standard_deviation"
                            ]
                        }
                    },
                ],
            },
            {
                "name": "pressure-features",
                "options": [
                    {"pick": {"features.sensors.pressure": ["mean"]}},
                    {
                        "pick": {
                            "features.sensors.pressure": ["standard_deviation"]
                        }
                    },
                ],
            },
        ],
    }

    trials = generate_trials(document)

    assert len(trials) == 6
    assert [trial["features"]["sensors"] for trial in trials] == [
        {"accelerometer": ["mean"]},
        {"accelerometer": ["standard_deviation"]},
        {"accelerometer": ["mean"], "pressure": ["mean"]},
        {
            "accelerometer": ["mean"],
            "pressure": ["standard_deviation"],
        },
        {"accelerometer": ["standard_deviation"], "pressure": ["mean"]},
        {
            "accelerometer": ["standard_deviation"],
            "pressure": ["standard_deviation"],
        },
    ]


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


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ([], "non-empty array of strings"),
        (["mean", "mean"], "duplicate selections"),
        (["maximum"], "unknown value"),
    ],
)
def test_rejects_invalid_array_pick_selections(selection, message):
    document = {
        "default": {
            "features": {
                "sensors": {
                    "accelerometer": ["mean", "standard_deviation"],
                }
            }
        },
        "dimensions": [
            {
                "name": "accelerometer-features",
                "options": [
                    {
                        "pick": {
                            "features.sensors.accelerometer": selection,
                        }
                    }
                ],
            }
        ],
    }

    with pytest.raises(TrialParametersError, match=message):
        generate_trials(document)


@pytest.mark.parametrize("child_operation", ["set", "pick"])
def test_rejects_conflicting_nested_operations(child_operation):
    document = {
        "default": {
            "features": {
                "sensors": {
                    "accelerometer": ["mean", "standard_deviation"],
                }
            }
        },
        "dimensions": [
            {
                "name": "sensor-set",
                "options": [
                    {"pick": {"features.sensors": ["accelerometer"]}},
                ],
            },
            {
                "name": "feature-conflict",
                "options": [
                    {
                        child_operation: {
                            (
                                "features.sensors.accelerometer"
                                if child_operation == "set"
                                else "features.sensors"
                            ): (
                                ["mean"]
                                if child_operation == "set"
                                else ["accelerometer"]
                            )
                        }
                    }
                ],
            },
        ],
    }

    with pytest.raises(TrialParametersError, match="overlapping paths"):
        generate_trials(document)
