from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path


def test_hash_is_canonical_json_of_trial_only(config_factory):
    config = config_factory()
    expected = hashlib.sha256(
        json.dumps(
            config.trial.raw,
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
