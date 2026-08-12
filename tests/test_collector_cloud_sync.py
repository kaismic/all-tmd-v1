from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "aws"
    / "remote"
    / "sync-collector-sessions.py"
)
SPEC = importlib.util.spec_from_file_location("collector_cloud_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
collector_cloud_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector_cloud_sync)


def test_sync_downloads_eligible_confirmed_sessions_and_checkpoints(tmp_path, monkeypatch):
    output_dir = tmp_path / "downloaded_sessions"
    sessions = [
        {
            "participant_id": "participant_001",
            "session_id": "session-1",
            "s3_key": "raw/participant_001/session-1.json.gz",
            "sync_key": "0000000000001#session-1",
        },
        {
            "participant_id": "test_001",
            "session_id": "test-session",
            "s3_key": "raw/test_001/test-session.json.gz",
            "sync_key": "0000000000002#test-session",
        },
    ]
    queried_after = []

    def fake_query(table, after_sync_key):
        queried_after.append((table, after_sync_key))
        return sessions

    def fake_download(bucket, destination, item):
        assert bucket == "collector-bucket"
        assert destination == output_dir
        return item["session_id"] == "session-1"

    monkeypatch.setattr(collector_cloud_sync, "query_sessions", fake_query)
    monkeypatch.setattr(collector_cloud_sync, "download_session", fake_download)

    result = collector_cloud_sync.sync(
        "collector-bucket", "TransportSessions", output_dir
    )

    assert result == {
        "discovered_count": 2,
        "eligible_count": 1,
        "downloaded_count": 1,
    }
    assert queried_after == [("TransportSessions", "")]
    checkpoint = json.loads(
        (output_dir / ".download_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["last_sync_key"] == "0000000000002#test-session"
    assert checkpoint["source_bucket"] == "collector-bucket"
    assert checkpoint["source_table"] == "TransportSessions"


def test_existing_checkpoint_limits_next_query(tmp_path, monkeypatch):
    output_dir = tmp_path / "downloaded_sessions"
    output_dir.mkdir()
    checkpoint = {
        "version": collector_cloud_sync.CHECKPOINT_VERSION,
        "last_sync_key": "0000000000042#session-42",
        "source_bucket": "collector-bucket",
        "source_table": "TransportSessions",
        "source_index": collector_cloud_sync.INDEX_NAME,
    }
    (output_dir / ".download_checkpoint.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )
    queried_after = []

    def fake_query(table, after_sync_key):
        queried_after.append(after_sync_key)
        return []

    monkeypatch.setattr(collector_cloud_sync, "query_sessions", fake_query)

    result = collector_cloud_sync.sync(
        "collector-bucket", "TransportSessions", output_dir
    )

    assert queried_after == ["0000000000042#session-42"]
    assert result["downloaded_count"] == 0


def test_eligibility_rejects_test_participants_and_mismatched_keys():
    assert collector_cloud_sync.is_eligible(
        {
            "participant_id": "participant_123",
            "s3_key": "raw/participant_123/session.json.gz",
        }
    )
    assert not collector_cloud_sync.is_eligible(
        {
            "participant_id": "test_123",
            "s3_key": "raw/test_123/session.json.gz",
        }
    )
    assert not collector_cloud_sync.is_eligible(
        {
            "participant_id": "participant_123",
            "s3_key": "raw/participant_999/session.json.gz",
        }
    )
