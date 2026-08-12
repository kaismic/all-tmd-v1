from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from all_tmd.aws_bundle import create_run_bundle


GIT_SHA = "a" * 40
COLLECTOR_ARGUMENTS = {
    "collector_sessions_bucket": "transport-data-sessions-123456789012",
    "collector_sessions_table": "TransportSessions",
}


def test_full_bundle_generates_cartesian_trials_and_manifest(tmp_path):
    project = _project(tmp_path)
    output = tmp_path / "bundle"

    manifest = create_run_bundle(
        project,
        output,
        run_id="20260804T120000Z-aaaaaaaa-full",
        git_repository="https://example.test/all-tmd-v1.git",
        git_commit=GIT_SHA,
        ntfy_topic="test-topic",
        created_at=datetime(2026, 8, 4, 12, tzinfo=timezone.utc),
        **COLLECTOR_ARGUMENTS,
    )

    trials = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    assert len(trials) == 4
    assert manifest["trial_count"] == 4
    assert manifest["generated_trial_count"] == 4
    assert manifest["mode"] == "full"
    assert manifest["notifications"]["topic"] == "test-topic"
    assert manifest["created_at"] == "2026-08-04T12:00:00+00:00"
    assert manifest["collector_sessions"] == {
        "bucket": COLLECTOR_ARGUMENTS["collector_sessions_bucket"],
        "table": COLLECTOR_ARGUMENTS["collector_sessions_table"],
    }
    for name, expected_digest in manifest["config_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == (
            expected_digest
        )


def test_smoke_bundle_uses_first_trial_and_one_optuna_evaluation(tmp_path):
    project = _project(tmp_path)
    output = tmp_path / "smoke"

    manifest = create_run_bundle(
        project,
        output,
        run_id="smoke-1",
        git_repository="https://example.test/all-tmd-v1.git",
        git_commit=GIT_SHA,
        mode="smoke",
        **COLLECTOR_ARGUMENTS,
    )

    trials = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    assert len(trials) == 1
    assert trials[0]["training"]["optuna_trials"] == 1
    assert manifest["trial_count"] == 1
    assert manifest["generated_trial_count"] == 4


@pytest.mark.parametrize("run_id", ["", "bad/id", "-starts-with-hyphen"])
def test_bundle_rejects_unsafe_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        create_run_bundle(
            _project(tmp_path),
            tmp_path / "bundle",
            run_id=run_id,
            git_repository="https://example.test/all-tmd-v1.git",
            git_commit=GIT_SHA,
            **COLLECTOR_ARGUMENTS,
        )


def test_bundle_never_contains_notification_token(tmp_path):
    output = tmp_path / "bundle"
    create_run_bundle(
        _project(tmp_path),
        output,
        run_id="safe-run",
        git_repository="https://example.test/all-tmd-v1.git",
        git_commit=GIT_SHA,
        ntfy_token_parameter="/private/token",
        **COLLECTOR_ARGUMENTS,
    )

    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    )
    assert "/private/token" in combined
    assert "NTFY_TOKEN=" not in combined


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "model.config.yaml").write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )
    parameters = {
        "default": {
            "features": {
                "default_window_seconds": 10,
                "default_step_seconds": 5,
                "sensors": {"accelerometer": ["mean", "range"]},
            },
            "training": {"optuna_trials": 45},
        },
        "dimensions": [
            {
                "name": "window",
                "options": [
                    {
                        "set": {
                            "features.default_window_seconds": window,
                            "features.default_step_seconds": step,
                        }
                    }
                    for window, step in ((10, 5), (20, 10), (30, 15), (60, 30))
                ],
            }
        ],
    }
    (project / "trial-parameters.json").write_text(
        json.dumps(parameters),
        encoding="utf-8",
    )
    return project
