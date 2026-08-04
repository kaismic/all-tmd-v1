from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("domain", ["us-tmd", "collector"])
def test_window_rejects_internal_gap_for_every_domain(config_factory, domain):
    config = config_factory(
        minimum_sampling_rate={"accelerometer": 4},
        maximum_sample_interval_ms=500,
    )
    events = _events(domain, "gapped", [0, 100, 800, 900])

    assert feature_frame(events, config).empty


def test_window_accepts_gap_equal_to_threshold(config_factory):
    config = config_factory(
        minimum_sampling_rate={"accelerometer": 2},
        maximum_sample_interval_ms=500,
    )

    assert len(feature_frame(_events("us-tmd", "source", [0, 500]), config)) == 1


def test_null_maximum_sample_interval_disables_continuity_check(config_factory):
    config = config_factory(
        minimum_sampling_rate={"accelerometer": 2},
        maximum_sample_interval_ms=None,
    )

    assert len(feature_frame(_events("us-tmd", "source", [0, 900]), config)) == 1


@pytest.mark.parametrize(
    "middle_timestamps",
    [
        [1600, 1900],
        [1000, 1400],
    ],
)
def test_boundary_gap_rejects_only_affected_window(
    config_factory,
    middle_timestamps,
):
    config = config_factory(
        minimum_sampling_rate={"accelerometer": 2},
        maximum_sample_interval_ms=500,
    )
    timestamps = [0, 500, *middle_timestamps, 2000, 2500]

    features = feature_frame(_events("collector", "session", timestamps), config)

    assert features["window_start_ms"].tolist() == [0, 2000]


def test_continuity_is_checked_per_sensor(config_factory):
    config = config_factory(
        sensors={"accelerometer": ["mean"], "pressure": ["range"]},
        minimum_sampling_rate={"accelerometer": 4, "pressure": 2},
        maximum_sample_interval_ms=500,
    )
    events = _events("us-tmd", "source", [0, 250, 500, 750])
    events["p"] = [1000.0, np.nan, np.nan, 1001.0]

    assert feature_frame(events, config).empty


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


def test_feature_policy_rebuilds_legacy_and_changed_cache(config_factory):
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
    _events("collector", "collector", [0, 1000]).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )
    output = build_features(config)["collector"]
    policy_path = output / "feature-policy.json"

    legacy_sentinel = output / "legacy-sentinel"
    legacy_sentinel.write_text("", encoding="utf-8")
    policy_path.unlink()
    build_features(config)
    assert not legacy_sentinel.exists()

    changed_sentinel = output / "changed-sentinel"
    changed_sentinel.write_text("", encoding="utf-8")
    changed_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    changed_policy["maximum_sample_interval_ms"] = 123
    policy_path.write_text(json.dumps(changed_policy), encoding="utf-8")
    build_features(config)
    assert not changed_sentinel.exists()

    corrupt_sentinel = output / "corrupt-sentinel"
    corrupt_sentinel.write_text("", encoding="utf-8")
    policy_path.write_text("not json", encoding="utf-8")
    build_features(config)
    assert not corrupt_sentinel.exists()

    matching_sentinel = output / "matching-sentinel"
    matching_sentinel.write_text("", encoding="utf-8")
    build_features(config)
    assert matching_sentinel.exists()


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
