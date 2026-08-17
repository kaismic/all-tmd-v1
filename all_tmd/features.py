from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import kurtosis


VECTOR_SENSOR_COLUMNS = {
    "accelerometer": ("ax", "ay", "az"),
    "gyroscope": ("gx", "gy", "gz"),
    "magnetometer": ("mx", "my", "mz"),
}
AGGREGATION_ALIASES = {
    "delta from session baseline": "delta_from_session_baseline",
    "delta from window start": "delta_from_window_start",
    "iqr": "interquartile_range",
    "max": "maximum",
    "min": "minimum",
    "std": "standard_deviation",
    "standard deviation": "standard_deviation",
    "var": "variance",
    "dominant frequency": "dominant_frequency_hz",
    "mean absolute jerk": "mean_absolute_jerk",
    "spectral entropy": "spectral_entropy",
    "spectral energy": "spectral_energy",
}
SUPPORTED_AGGREGATIONS = {
    "minimum",
    "maximum",
    "mean",
    "range",
    "variance",
    "standard_deviation",
    "kurtosis",
    "interquartile_range",
    "delta_from_window_start",
    "delta_from_session_baseline",
    "mean_absolute_deviation",
    "mean_absolute_jerk",
    "jerk_standard_deviation",
    "spectral_energy",
    "dominant_frequency_hz",
    "spectral_entropy",
    "mean_axis_correlation",
}


def normalize_aggregation_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    canonical = AGGREGATION_ALIASES.get(normalized, normalized.replace(" ", "_"))
    if canonical not in SUPPORTED_AGGREGATIONS:
        supported = ", ".join(sorted(SUPPORTED_AGGREGATIONS))
        raise ValueError(f"Unsupported aggregation '{name}'. Supported: {supported}")
    return canonical


def aggregate(values: np.ndarray, session_baseline: float | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty sensor window")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    kurtosis_value = float(kurtosis(values, fisher=True, bias=False))
    if not np.isfinite(kurtosis_value):
        kurtosis_value = 0.0
    baseline = values[0] if session_baseline is None else session_baseline
    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": float(np.mean(values)),
        "range": maximum - minimum,
        "variance": float(np.var(values, ddof=0)),
        "standard_deviation": float(np.std(values, ddof=0)),
        "kurtosis": kurtosis_value,
        "interquartile_range": float(q3 - q1),
        "delta_from_window_start": float(values[-1] - values[0]),
        "delta_from_session_baseline": float(np.mean(values) - baseline),
        "mean_absolute_deviation": float(np.mean(np.abs(values - np.mean(values)))),
    }


def sensor_series(sensor: str, rows) -> np.ndarray:
    if sensor == "pressure":
        values = rows["p"].to_numpy(dtype=np.float64)
        return values[np.isfinite(values)]
    columns = VECTOR_SENSOR_COLUMNS[sensor]
    values = rows.loc[:, columns].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    return np.linalg.norm(values, axis=1) if values.size else np.array([], dtype=np.float64)


def ordered_sensor_features(
    sensor: str,
    rows,
    aggregations: Sequence[str],
    session_baseline: float | None = None,
) -> list[float]:
    timestamps, values = sensor_samples(sensor, rows)
    stats = aggregate(values, session_baseline=session_baseline)
    stats.update(_temporal_features(timestamps, values))
    stats["mean_axis_correlation"] = _mean_axis_correlation(sensor, rows)
    return [stats[name] for name in aggregations]


def sensor_samples(sensor: str, rows) -> tuple[np.ndarray, np.ndarray]:
    if sensor == "pressure":
        valid = np.isfinite(rows["p"].to_numpy(dtype=np.float64))
        values = rows.loc[valid, "p"].to_numpy(dtype=np.float64)
    else:
        columns = VECTOR_SENSOR_COLUMNS[sensor]
        vectors = rows.loc[:, columns].to_numpy(dtype=np.float64)
        valid = np.isfinite(vectors).all(axis=1)
        values = np.linalg.norm(vectors[valid], axis=1)
    timestamps = rows.loc[valid, "timestamp_ms"].to_numpy(dtype=np.float64)
    if not timestamps.size:
        return timestamps, values
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    values = values[order]
    unique = np.concatenate(([True], np.diff(timestamps) > 0))
    return timestamps[unique], values[unique]


def _temporal_features(
    timestamps_ms: np.ndarray,
    values: np.ndarray,
) -> dict[str, float]:
    zero = {
        "mean_absolute_jerk": 0.0,
        "jerk_standard_deviation": 0.0,
        "spectral_energy": 0.0,
        "dominant_frequency_hz": 0.0,
        "spectral_entropy": 0.0,
    }
    if len(values) < 2:
        return zero
    seconds = timestamps_ms / 1000.0
    intervals = np.diff(seconds)
    positive = intervals > 0
    if not positive.any():
        return zero
    jerk = np.diff(values)[positive] / intervals[positive]
    result = {
        **zero,
        "mean_absolute_jerk": float(np.mean(np.abs(jerk))),
        "jerk_standard_deviation": float(np.std(jerk, ddof=0)),
    }
    if len(values) < 4:
        return result
    interval = float(np.median(intervals[positive]))
    if not np.isfinite(interval) or interval <= 0:
        return result
    uniform_seconds = np.arange(seconds[0], seconds[-1] + interval / 2, interval)
    if len(uniform_seconds) < 4:
        return result
    uniform_values = np.interp(uniform_seconds, seconds, values)
    dynamic = uniform_values - np.mean(uniform_values)
    spectrum = np.fft.rfft(dynamic)
    power = np.square(np.abs(spectrum)) / (len(dynamic) ** 2)
    frequencies = np.fft.rfftfreq(len(dynamic), d=interval)
    if power.size:
        power[0] = 0.0
    total_power = float(power.sum())
    result["spectral_energy"] = total_power
    if total_power > 0 and len(power) > 1:
        dominant = int(np.argmax(power[1:]) + 1)
        result["dominant_frequency_hz"] = float(frequencies[dominant])
        probabilities = power[1:] / total_power
        positive_probabilities = probabilities[probabilities > 0]
        entropy = -float(np.sum(positive_probabilities * np.log(positive_probabilities)))
        normalizer = np.log(len(probabilities)) if len(probabilities) > 1 else 1.0
        result["spectral_entropy"] = entropy / normalizer
    return result


def _mean_axis_correlation(sensor: str, rows) -> float:
    if sensor == "pressure":
        return 0.0
    values = rows.loc[:, VECTOR_SENSOR_COLUMNS[sensor]].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 2:
        return 0.0
    correlations = np.corrcoef(values, rowvar=False)
    pairs = np.abs(correlations[np.triu_indices(3, k=1)])
    pairs = pairs[np.isfinite(pairs)]
    return float(np.mean(pairs)) if pairs.size else 0.0
