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
    stats = aggregate(sensor_series(sensor, rows), session_baseline=session_baseline)
    return [stats[name] for name in aggregations]
