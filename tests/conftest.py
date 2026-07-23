from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from all_tmd.config import PipelineConfig


@pytest.fixture
def config_factory(tmp_path):
    def factory(
        *,
        train_dataset: str = "us-tmd",
        sensors: dict[str, list[str]] | None = None,
        minimum_sampling_rate: dict[str, float] | None = None,
        minimum_trip_seconds: int = 0,
        collector_max_sample_interval_ms: int | None = None,
        calibration_fraction: float = 0.5,
    ) -> PipelineConfig:
        sensors = sensors or {"accelerometer": ["mean"]}
        minimum_sampling_rate = minimum_sampling_rate or {
            sensor: 1 for sensor in sensors
        }
        data_root = tmp_path / "data"
        config_data = {
            "schema_version": 1,
            "sources": {
                "us-tmd": {
                    "input_dir": str(data_root / "us"),
                    "include_globs": ["**/*.csv"],
                    "chunk_rows": 100,
                },
                "nor-tmd": {
                    "input_dir": str(data_root / "nor"),
                    "include_globs": ["**/*.csv"],
                    "chunk_rows": 100,
                },
                "collector": {
                    "input_dir": str(data_root / "collector"),
                    "include_globs": ["**/*.json", "**/*.json.gz"],
                },
            },
            "dataset": {
                "work_dir": str(data_root / "work"),
                "minimum_trip_seconds": minimum_trip_seconds,
                "maximum_trip_seconds": 28_800,
                "collector_max_sample_interval_ms": collector_max_sample_interval_ms,
            },
            "minimum_sampling_rate": minimum_sampling_rate,
            "training": {
                "n_jobs": 1,
                "timeout_seconds": None,
                "xgboost_device": "cpu",
            },
            "mlflow": {
                "enabled": False,
                "experiment_name": "test",
                "tracking_uri": None,
            },
        }
        trial = {
            "train_dataset": train_dataset,
            "labels": {"bus": 0, "car": 1, "train": 2},
            "features": {
                "default_window_seconds": 1,
                "default_step_seconds": 1,
                "sensors": sensors,
            },
            "training": {
                "random_seed": 42,
                "optuna_trials": 1,
                "model_families": ["random_forest"],
                "calibration_fraction": calibration_fraction,
            },
        }
        config_path = tmp_path / f"{train_dataset}.yaml"
        trials_path = tmp_path / f"{train_dataset}.json"
        config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
        trials_path.write_text(json.dumps([trial]), encoding="utf-8")
        return PipelineConfig.from_files(config_path, trials_path)

    return factory
