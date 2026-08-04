from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from all_tmd.config import PipelineConfig
from all_tmd.features import VECTOR_SENSOR_COLUMNS, ordered_sensor_features, sensor_series
from all_tmd.progress import progress


METADATA_COLUMNS = [
    "domain",
    "participant_id",
    "device_id",
    "session_id",
    "trip_id",
    "group_id",
    "vehicle_type",
    "label",
    "phone_position",
    "window_start_ms",
    "window_end_ms",
]
FEATURE_BUCKETS = 64
FEATURE_CHECKPOINT = "checkpoint.json"
FEATURE_POLICY = "feature-policy.json"
FEATURE_POLICY_SCHEMA_VERSION = 1


def build_features(config: PipelineConfig) -> dict[str, Path]:
    source = config.trial.train_dataset
    return {
        source: _build_source_features(config, source, incremental=False),
        "collector": _build_source_features(config, "collector", incremental=True),
    }


def _build_source_features(
    config: PipelineConfig,
    source_name: str,
    *,
    incremental: bool,
) -> Path:
    run_dir = config.run_dir()
    event_dir = run_dir / "events" / source_name
    if not event_dir.exists() or not any(event_dir.glob("part-*.parquet")):
        raise FileNotFoundError(f"Run ingestion before features: {event_dir}")
    if not incremental and not (event_dir / "_SUCCESS").exists():
        raise RuntimeError(
            f"Training event dataset is incomplete (missing _SUCCESS): {event_dir}"
        )
    output_dir = run_dir / "features" / source_name
    expected_policy = _feature_policy(config)
    if output_dir.exists() and not _feature_policy_matches(
        output_dir / FEATURE_POLICY,
        expected_policy,
    ):
        progress(f"Rebuilding features for changed policy: {output_dir}")
        shutil.rmtree(output_dir)
    success_path = output_dir / "_SUCCESS"
    checkpoint_path = output_dir / FEATURE_CHECKPOINT
    if not incremental and success_path.exists():
        progress(f"Training features already complete: {output_dir}")
        return output_dir
    if not incremental and output_dir.exists():
        progress(f"Removing incomplete training feature dataset: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_feature_policy(output_dir / FEATURE_POLICY, expected_policy)

    processed = _read_checkpoint(checkpoint_path) if incremental else set()
    if incremental:
        processed.update(_existing_feature_session_ids(output_dir))
    pending_ids = _event_session_ids(event_dir) - processed
    if not pending_ids:
        if incremental and any(output_dir.glob("part-*.parquet")):
            _write_checkpoint(checkpoint_path, processed)
            progress(f"No new collector sessions require feature extraction: {output_dir}")
            return output_dir
        if not incremental and success_path.exists():
            return output_dir
        raise ValueError(f"No unprocessed event sessions found for {source_name}")

    temp_dir = output_dir / ".feature-build-tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    try:
        _bucket_pending_events(
            event_dir,
            temp_dir,
            pending_ids,
            _required_event_columns(config),
        )
        next_part = _next_part_number(output_dir)
        total_rows = 0
        completed_ids: set[str] = set()
        for bucket_id in range(FEATURE_BUCKETS):
            bucket_dir = temp_dir / f"bucket-{bucket_id:04d}"
            parts = sorted(bucket_dir.glob("part-*.parquet"))
            if not parts:
                continue
            events = pd.concat(
                (pd.read_parquet(part) for part in parts),
                ignore_index=True,
            ).sort_values(["session_id", "timestamp_ms"])
            bucket_ids = set(events["session_id"].astype(str))
            features = feature_frame(events, config)
            if not features.empty:
                output_path = output_dir / f"part-{next_part:06d}.parquet"
                features.to_parquet(output_path, index=False)
                next_part += 1
                total_rows += len(features)
            completed_ids.update(bucket_ids)
            if incremental:
                processed.update(bucket_ids)
                _write_checkpoint(checkpoint_path, processed)
            progress(
                f"Feature bucket complete: source={source_name}, "
                f"bucket={bucket_id + 1}/{FEATURE_BUCKETS}, rows={len(features):,}"
            )

        if not completed_ids:
            raise ValueError(f"No event sessions could be processed for {source_name}")
        if not any(output_dir.glob("part-*.parquet")):
            raise ValueError(f"No feature rows produced for {source_name}")
        if not incremental:
            success_path.write_text("", encoding="utf-8")
        progress(
            f"Feature extraction complete: source={source_name}, "
            f"new_rows={total_rows:,}, sessions={len(completed_ids):,}, output={output_dir}"
        )
        return output_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def feature_frame(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    feature_config = config.trial.features
    minimum_samples = config.trial.minimum_samples(config.minimum_sampling_rate)
    window_ms = feature_config.default_window_seconds * 1000
    step_ms = feature_config.default_step_seconds * 1000
    rows: list[dict[str, Any]] = []

    for _, session in frame.groupby("session_id", sort=False):
        session = session.sort_values("timestamp_ms")
        start = int(session["timestamp_ms"].min())
        end = int(session["timestamp_ms"].max())
        duration = (end - start) / 1000.0
        if not (
            config.dataset.minimum_trip_seconds
            <= duration
            <= config.dataset.maximum_trip_seconds
        ):
            continue
        baselines = _session_baselines(
            session,
            set(feature_config.sensors),
        )
        timestamps = session["timestamp_ms"].to_numpy(dtype=np.int64)
        positive_diffs = np.diff(timestamps)
        positive_diffs = positive_diffs[positive_diffs > 0]
        sample_interval = (
            int(np.median(positive_diffs)) if positive_diffs.size else step_ms
        )
        exclusive_end = end + sample_interval
        cursor = start
        while cursor + window_ms <= exclusive_end:
            window = session[
                (session["timestamp_ms"] >= cursor)
                & (session["timestamp_ms"] < cursor + window_ms)
            ]
            if _meets_window_quality_requirements(
                window,
                minimum_samples,
                config.dataset.maximum_sample_interval_ms,
                cursor,
                cursor + window_ms,
            ):
                rows.append(
                    _window_features(
                        window,
                        config,
                        baselines,
                        cursor,
                        cursor + window_ms,
                    )
                )
            cursor += step_ms
    return pd.DataFrame(rows)


def _session_baselines(
    session: pd.DataFrame,
    configured_sensors: set[str],
) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for sensor in configured_sensors.intersection({"pressure", "magnetometer"}):
        values = sensor_series(sensor, session)
        if values.size:
            baselines[sensor] = (
                float(values[0])
                if sensor == "pressure"
                else float(np.median(values))
            )
    return baselines


def _window_features(
    window: pd.DataFrame,
    config: PipelineConfig,
    baselines: dict[str, float],
    window_start_ms: int,
    window_end_ms: int,
) -> dict[str, Any]:
    first = window.iloc[0]
    row: dict[str, Any] = {
        column: first[column]
        for column in METADATA_COLUMNS
        if column not in {"window_start_ms", "window_end_ms"}
    }
    row["label"] = int(first["label"])
    row["window_start_ms"] = int(window_start_ms)
    row["window_end_ms"] = int(window_end_ms)
    for sensor, aggregations in config.trial.features.sensors.items():
        values = ordered_sensor_features(
            sensor,
            window,
            aggregations,
            session_baseline=baselines.get(sensor),
        )
        for aggregation, value in zip(aggregations, values):
            row[f"{sensor}#{aggregation}"] = np.float32(value)
    return row


def _meets_window_quality_requirements(
    window: pd.DataFrame,
    minimum_samples: dict[str, int],
    maximum_sample_interval_ms: int | None,
    window_start_ms: int,
    window_end_ms: int,
) -> bool:
    for sensor, required in minimum_samples.items():
        if sensor == "pressure":
            valid = window["p"].notna()
        else:
            valid = window.loc[:, VECTOR_SENSOR_COLUMNS[sensor]].notna().all(axis=1)
        timestamps = np.sort(
            window.loc[valid, "timestamp_ms"].to_numpy(dtype=np.int64)
        )
        if timestamps.size < required:
            return False
        if maximum_sample_interval_ms is not None:
            bounded_timestamps = np.concatenate(
                (
                    np.array([window_start_ms], dtype=np.int64),
                    timestamps,
                    np.array([window_end_ms], dtype=np.int64),
                )
            )
            if (
                np.diff(bounded_timestamps).max(initial=0)
                > maximum_sample_interval_ms
            ):
                return False
    return True


def _feature_policy(config: PipelineConfig) -> dict[str, Any]:
    configured_rates = {
        sensor: config.minimum_sampling_rate[sensor]
        for sensor in sorted(config.trial.features.sensors)
    }
    return {
        "schema_version": FEATURE_POLICY_SCHEMA_VERSION,
        "trial_config_hash": config.config_hash,
        "minimum_sampling_rate": configured_rates,
        "minimum_trip_seconds": config.dataset.minimum_trip_seconds,
        "maximum_trip_seconds": config.dataset.maximum_trip_seconds,
        "maximum_sample_interval_ms": config.dataset.maximum_sample_interval_ms,
    }


def _feature_policy_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _write_feature_policy(path: Path, policy: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(policy, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _required_event_columns(config: PipelineConfig) -> list[str]:
    columns = {
        "domain",
        "participant_id",
        "device_id",
        "session_id",
        "trip_id",
        "group_id",
        "vehicle_type",
        "label",
        "phone_position",
        "timestamp_ms",
    }
    for sensor in config.trial.features.sensors:
        if sensor == "pressure":
            columns.add("p")
        else:
            columns.update(VECTOR_SENSOR_COLUMNS[sensor])
    return sorted(columns)


def _bucket_pending_events(
    event_dir: Path,
    temp_dir: Path,
    pending_ids: set[str],
    columns: list[str],
) -> None:
    for part_number, part in enumerate(sorted(event_dir.glob("part-*.parquet"))):
        frame = pd.read_parquet(part, columns=columns)
        frame = frame[frame["session_id"].astype(str).isin(pending_ids)]
        if frame.empty:
            continue
        bucket_ids = (
            pd.util.hash_pandas_object(frame["session_id"].astype("string"), index=False)
            % FEATURE_BUCKETS
        ).astype("int64")
        for bucket_id, bucket in frame.groupby(bucket_ids, sort=False):
            bucket_dir = temp_dir / f"bucket-{int(bucket_id):04d}"
            bucket_dir.mkdir(exist_ok=True)
            bucket.to_parquet(
                bucket_dir / f"part-{part_number:06d}.parquet",
                index=False,
            )


def _event_session_ids(event_dir: Path) -> set[str]:
    result: set[str] = set()
    for part in sorted(event_dir.glob("part-*.parquet")):
        frame = pd.read_parquet(part, columns=["session_id"])
        result.update(frame["session_id"].dropna().astype(str))
    return result


def _existing_feature_session_ids(output_dir: Path) -> set[str]:
    result: set[str] = set()
    for part in sorted(output_dir.glob("part-*.parquet")):
        frame = pd.read_parquet(part, columns=["session_id"])
        result.update(frame["session_id"].dropna().astype(str))
    return result


def _read_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Invalid feature checkpoint: {path}")
    return set(values)


def _write_checkpoint(path: Path, values: set[str]) -> None:
    path.write_text(json.dumps(sorted(values), indent=2) + "\n", encoding="utf-8")


def _next_part_number(output_dir: Path) -> int:
    values: list[int] = []
    for part in output_dir.glob("part-*.parquet"):
        try:
            values.append(int(part.stem.removeprefix("part-")))
        except ValueError:
            continue
    return max(values, default=-1) + 1
