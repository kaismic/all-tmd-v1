from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from all_tmd.deployment import (
    collector_session_ids_from_log,
    select_balanced_fixtures,
)


def test_collector_session_ids_from_log_recovers_one_scan(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "Collector ingest scan starting: files=2, checkpoint_sessions=0",
                "Collector ingest file 1/2: status=ingested, "
                "path=/raw/00000000-0000-0000-0000-000000000001.json.gz",
                "Collector ingest file 2/2: status=ingested, "
                "path=/raw/00000000-0000-0000-0000-000000000002.json.gz",
            ]
        ),
        encoding="utf-8",
    )

    assert collector_session_ids_from_log(log) == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]


def test_collector_session_ids_from_log_rejects_incomplete_scan(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "Collector ingest scan starting: files=2\n"
        "Collector ingest file 1/2: path=/raw/"
        "00000000-0000-0000-0000-000000000001.json.gz\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Recovered 1 of 2"):
        collector_session_ids_from_log(log)


def test_select_balanced_fixtures_is_deterministic():
    frame = pd.DataFrame(
        [
            {
                "label": label,
                "session_id": f"s-{label}-{index}",
                "window_start_ms": index,
                "window_end_ms": index + 60,
            }
            for label in range(3)
            for index in range(5)
        ]
    )

    selected = select_balanced_fixtures(frame, fixtures_per_class=2)

    assert selected.groupby("label").size().to_dict() == {0: 2, 1: 2, 2: 2}
    assert selected["session_id"].tolist() == [
        "s-0-0",
        "s-0-4",
        "s-1-0",
        "s-1-4",
        "s-2-0",
        "s-2-4",
    ]
