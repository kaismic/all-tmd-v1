from __future__ import annotations

import json
from pathlib import Path

import pytest

from all_tmd.config import PipelineConfig
from all_tmd.trial_generator import generate_trials


PROJECT_ROOT = Path(__file__).parents[1]
PLANS_ROOT = PROJECT_ROOT / "recommended-trial-plans"
PLAN_NAMES = (
    "01-participant-baseline",
    "02-weighting-ablation",
    "03-duration-ablation",
    "04-multiscale-frequency-ablation",
    "05-final-model-search",
)


@pytest.mark.parametrize("plan_name", PLAN_NAMES)
def test_recommended_plan_is_current_named_and_valid(plan_name):
    parameters_path = PLANS_ROOT / f"{plan_name}.trial-parameters.json"
    trials_path = PLANS_ROOT / f"{plan_name}.trials.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    committed_trials = json.loads(trials_path.read_text(encoding="utf-8"))

    assert committed_trials == generate_trials(parameters)
    run_names = [trial.get("run_name") for trial in committed_trials]
    assert all(
        isinstance(run_name, str) and run_name.startswith(plan_name[:2] + "-")
        for run_name in run_names
    )
    assert len(run_names) == len(set(run_names))

    for trial_index in range(len(committed_trials)):
        config = PipelineConfig.from_files(
            PROJECT_ROOT / "model.config.yaml",
            trials_path,
            trial_index,
        )
        assert config.trial.run_name == run_names[trial_index]
