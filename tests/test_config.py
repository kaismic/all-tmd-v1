from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest


def test_hash_is_canonical_json_of_trial_without_training(config_factory):
    config = config_factory()
    expected = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in config.trial.raw.items()
                if key != "training"
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert config.config_hash == expected
    changed_global_config = replace(
        config,
        dataset=replace(config.dataset, work_dir=Path("a-different-work-dir")),
    )
    assert changed_global_config.config_hash == config.config_hash


def test_training_fields_do_not_affect_hash_or_cause_collision(config_factory):
    config = config_factory()
    changed_training = replace(
        config,
        trial=replace(
            config.trial,
            training=replace(
                config.trial.training,
                random_seed=123,
                optuna_trials=25,
            ),
            raw={
                **config.trial.raw,
                "training": {
                    **config.trial.raw["training"],
                    "random_seed": 123,
                    "optuna_trials": 25,
                },
            },
        ),
    )

    assert changed_training.config_hash == config.config_hash
    original_run_dir = config.run_dir()
    assert changed_training.run_dir() == original_run_dir
    saved_trial = json.loads(
        (original_run_dir / "trial.json").read_text(encoding="utf-8")
    )
    assert saved_trial["training"]["random_seed"] == 123
    assert saved_trial["training"]["optuna_trials"] == 25


def test_scalar_calibration_fraction_applies_to_every_mode(config_factory):
    config = config_factory(calibration_fraction=0.5)

    assert config.trial.training.calibration_fraction == {
        "bus": 0.5,
        "car": 0.5,
        "train": 0.5,
    }


def test_calibration_fractions_are_configurable_per_mode(config_factory):
    config = config_factory(
        calibration_fraction={"bus": 0.8, "car": 0.4, "train": 0.4}
    )

    assert config.trial.training.calibration_fraction == {
        "bus": 0.8,
        "car": 0.4,
        "train": 0.4,
    }


@pytest.mark.parametrize(
    ("calibration_fraction", "message"),
    [
        ({"bus": 0.5, "car": 0.5}, "missing transport mode.*train"),
        (
            {"bus": 0.5, "car": 0.5, "train": 0.5, "tram": 0.5},
            "unknown transport mode.*tram",
        ),
        (
            {"bus": 0.5, "BUS": 0.5, "car": 0.5, "train": 0.5},
            "duplicate transport modes",
        ),
    ],
)
def test_calibration_fraction_modes_must_exactly_match_labels(
    config_factory,
    calibration_fraction,
    message,
):
    with pytest.raises(ValueError, match=message):
        config_factory(calibration_fraction=calibration_fraction)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), float("inf")])
def test_calibration_fractions_must_be_finite_and_between_zero_and_one(
    config_factory,
    fraction,
):
    with pytest.raises(
        ValueError,
        match=(
            "training.calibration_fraction.bus must be a finite number "
            "between 0 and 1"
        ),
    ):
        config_factory(
            calibration_fraction={
                "bus": fraction,
                "car": 0.5,
                "train": 0.5,
            }
        )


def test_calibration_fraction_must_be_numeric(config_factory):
    with pytest.raises(
        ValueError,
        match="training.calibration_fraction.bus must be a number between 0 and 1",
    ):
        config_factory(
            calibration_fraction={"bus": None, "car": 0.5, "train": 0.5}
        )


def test_minimum_samples_are_calculated_per_sensor(config_factory):
    config = config_factory(
        sensors={"accelerometer": ["mean"], "gyroscope": ["range"]},
        collector_minimum_sampling_rate={"accelerometer": 30, "gyroscope": 4},
    )
    assert config.trial.minimum_samples(
        config.collector_minimum_sampling_rate
    ) == {
        "accelerometer": 30,
        "gyroscope": 4,
    }


def test_fractional_sample_requirement_rounds_up(config_factory):
    config = config_factory(
        collector_minimum_sampling_rate={"accelerometer": 1.5}
    )
    assert config.trial.minimum_samples(
        config.collector_minimum_sampling_rate
    ) == {
        "accelerometer": 2,
    }


def test_maximum_sample_interval_must_be_positive_or_null(config_factory):
    disabled = config_factory(collector_max_sample_interval_ms=None)
    enabled = config_factory(collector_max_sample_interval_ms=500)
    assert disabled.dataset.collector_max_sample_interval_ms is None
    assert enabled.dataset.collector_max_sample_interval_ms == 500
    with pytest.raises(
        ValueError,
        match=(
            "dataset.collector_max_sample_interval_ms must be greater than "
            "zero or null"
        ),
    ):
        config_factory(collector_max_sample_interval_ms=0)


def test_generic_sampling_quality_keys_are_rejected(config_factory):
    with pytest.raises(
        ValueError,
        match=(
            "minimum_sampling_rate was renamed to "
            "collector_minimum_sampling_rate"
        ),
    ):
        config_factory(generic_minimum_sampling_rate={"accelerometer": 1})
    with pytest.raises(
        ValueError,
        match=(
            "dataset.maximum_sample_interval_ms was renamed to "
            "dataset.collector_max_sample_interval_ms"
        ),
    ):
        config_factory(generic_maximum_sample_interval_ms=500)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_collector_minimum_sampling_rates_must_be_finite_and_positive(
    config_factory,
    rate,
):
    with pytest.raises(
        ValueError,
        match=(
            "collector_minimum_sampling_rate must contain finite positive "
            "values for: accelerometer"
        ),
    ):
        config_factory(
            collector_minimum_sampling_rate={"accelerometer": rate}
        )


def test_collector_minimum_sampling_rates_cover_selected_sensors(config_factory):
    with pytest.raises(
        ValueError,
        match=(
            "collector_minimum_sampling_rate is missing configured sensor.*pressure"
        ),
    ):
        config_factory(
            sensors={"accelerometer": ["mean"], "pressure": ["range"]},
            collector_minimum_sampling_rate={"accelerometer": 30},
        )


def test_nor_source_accepts_a_file_path(config_factory, tmp_path):
    config = config_factory(train_dataset="nor-tmd")
    source = config.sources.training["nor-tmd"]
    assert source.input_path == tmp_path / "data" / "nor"
