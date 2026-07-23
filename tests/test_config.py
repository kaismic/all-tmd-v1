from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path


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


def test_minimum_samples_are_calculated_per_sensor(config_factory):
    config = config_factory(
        sensors={"accelerometer": ["mean"], "gyroscope": ["range"]},
        minimum_sampling_rate={"accelerometer": 30, "gyroscope": 4},
    )
    assert config.trial.minimum_samples(config.minimum_sampling_rate) == {
        "accelerometer": 30,
        "gyroscope": 4,
    }


def test_fractional_sample_requirement_rounds_up(config_factory):
    config = config_factory(minimum_sampling_rate={"accelerometer": 1.5})
    assert config.trial.minimum_samples(config.minimum_sampling_rate) == {
        "accelerometer": 2,
    }


def test_nor_source_accepts_a_file_path(config_factory, tmp_path):
    config = config_factory(train_dataset="nor-tmd")
    source = config.sources.training["nor-tmd"]
    assert source.input_path == tmp_path / "data" / "nor"
