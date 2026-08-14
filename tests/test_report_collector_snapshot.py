import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "report-collector-snapshot.py"
SPEC = importlib.util.spec_from_file_location("report_collector_snapshot", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_format_duration_always_includes_hours_minutes_and_seconds():
    assert MODULE.format_duration(0) == "00:00:00"
    assert MODULE.format_duration(3155) == "00:52:35"
    assert MODULE.format_duration((52 * 60 + 35) * 60) == "52:35:00"


def test_main_reports_uploaded_snapshot_by_mode_and_participant(tmp_path, capsys):
    sessions = [
        {
            "session_id": "one",
            "participant_id": "participant_001",
            "vehicle_type": "train",
            "duration_seconds": 60,
            "sample_count": 120,
        },
        {
            "session_id": "two",
            "participant_id": "participant_002",
            "vehicle_type": "bus",
            "duration_seconds": 120,
            "sample_count": 240,
        },
    ]
    snapshot_path = tmp_path / "run-1" / "run" / "collector-snapshot.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "session_count": 2,
                "session_id_digest": MODULE.session_id_digest(sessions),
                "sessions": sessions,
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.main(["run-1", "--results-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert result == 0
    assert "Transport Mode Summary" in output
    assert "bus" in output and "00:02:00" in output
    assert "train" in output and "00:01:00" in output
    assert "participant_001" in output
    assert "participant_002" in output
    assert "Total" in output and "00:03:00" in output


def test_main_reconstructs_legacy_run_from_log_and_sidecars(tmp_path, capsys):
    log_path = tmp_path / "legacy-run" / "run" / "run.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "path=/data/downloaded_sessions/raw/participant_001/one.json.gz\n",
        encoding="utf-8",
    )
    sessions_dir = tmp_path / "downloaded_sessions"
    metadata_path = (
        sessions_dir / "raw" / "participant_001" / "one.json.gz.metadata.json"
    )
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "session_id": "one",
                "participant_id": "participant_001",
                "vehicle_type": "car",
                "trimmed_start_ms": 0,
                "trimmed_end_ms": 30000,
                "sample_count": 60,
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.main(
        [
            "legacy-run",
            "--results-root",
            str(tmp_path),
            "--sessions-dir",
            str(sessions_dir),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "legacy run.log plus sidecars" in output
    assert "car" in output
    assert "00:00:30" in output


def test_main_rejects_manifest_for_another_run(tmp_path, capsys):
    snapshot_path = tmp_path / "run-1" / "run" / "collector-snapshot.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "another-run",
                "session_count": 0,
                "session_id_digest": MODULE.session_id_digest([]),
                "sessions": [],
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.main(["run-1", "--results-root", str(tmp_path)])

    assert result == 1
    assert "does not match" in capsys.readouterr().err
