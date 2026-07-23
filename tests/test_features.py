from __future__ import annotations

import numpy as np
import pandas as pd

from all_tmd.ingest import EVENT_COLUMNS
from all_tmd.windowing import build_features, feature_frame


def test_window_requires_sampling_rate_times_window(config_factory):
    config = config_factory(
        sensors={"accelerometer": ["mean"]},
        minimum_sampling_rate={"accelerometer": 2},
    )
    enough = _events("us-tmd", "source", [0, 500])
    assert len(feature_frame(enough, config)) == 1
    assert feature_frame(enough.iloc[:1], config).empty


def test_features_are_incremental_per_source(config_factory):
    config = config_factory()
    run_dir = config.run_dir()
    source_dir = run_dir / "events" / "us-tmd"
    collector_dir = run_dir / "events" / "collector"
    source_dir.mkdir(parents=True)
    collector_dir.mkdir(parents=True)
    _events("us-tmd", "source", [0, 1000]).to_parquet(
        source_dir / "part-000000.parquet",
        index=False,
    )
    (source_dir / "_SUCCESS").write_text("", encoding="utf-8")
    _events("collector", "collector-1", [0, 1000]).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )

    outputs = build_features(config)
    assert (outputs["us-tmd"] / "_SUCCESS").exists()
    collector_parts = sorted(outputs["collector"].glob("part-*.parquet"))
    assert len(collector_parts) == 1

    build_features(config)
    assert sorted(outputs["collector"].glob("part-*.parquet")) == collector_parts

    _events("collector", "collector-2", [0, 1000]).to_parquet(
        collector_dir / "part-000001.parquet",
        index=False,
    )
    build_features(config)
    assert len(list(outputs["collector"].glob("part-*.parquet"))) == 2


def _events(domain: str, session_id: str, timestamps: list[int]) -> pd.DataFrame:
    rows = []
    for index, timestamp in enumerate(timestamps):
        row = {column: np.nan for column in EVENT_COLUMNS}
        row.update(
            {
                "domain": domain,
                "participant_id": f"p-{session_id}",
                "device_id": "device",
                "session_id": session_id,
                "trip_id": session_id,
                "group_id": f"p-{session_id}#device#{session_id}",
                "vehicle_type": "car",
                "label": 1,
                "phone_position": "pocket",
                "timestamp_ms": timestamp,
                "ax": index + 1,
                "ay": index + 2,
                "az": index + 3,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
