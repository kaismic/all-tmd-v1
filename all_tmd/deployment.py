from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from all_tmd.features import aggregate
from all_tmd.splits import frame_fingerprint


E821EDC_RUN_ID = "e821edccef3648d1be52848dd413f007"
E821EDC_SHORT_RUN_ID = E821EDC_RUN_ID[:7]
E821EDC_FEATURE_NAMES = (
    "accelerometer#mean",
    "accelerometer#standard_deviation",
    "accelerometer#range",
    "accelerometer#minimum",
    "gyroscope#mean",
    "gyroscope#standard_deviation",
    "gyroscope#range",
    "gyroscope#minimum",
    "magnetometer#standard_deviation",
    "magnetometer#range",
    "magnetometer#minimum",
)
LABEL_NAMES = {0: "bus", 1: "car", 2: "train"}


@dataclass(frozen=True)
class ExportPaths:
    model: Path
    metadata: Path
    fixtures: Path


def export_e821edc(
    *,
    model_path: Path,
    metrics_path: Path,
    split_manifest_path: Path,
    feature_cache: Path,
    collector_run_log: Path,
    output_dir: Path,
    fixtures_per_class: int = 32,
) -> ExportPaths:
    """Export and verify the selected run as an ONNX mobile bundle."""
    metrics = _read_json(metrics_path)
    manifest = _read_json(split_manifest_path)
    _validate_run_metadata(metrics)

    pipeline = joblib.load(model_path)
    _validate_pipeline(pipeline)
    model_onnx = _convert_pipeline(pipeline)

    frame, recovered_fingerprint = _load_run_frame(
        feature_cache,
        collector_session_ids_from_log(collector_run_log),
    )
    holdout = frame.loc[manifest["collector_holdout_indices"]].copy()
    _validate_recovered_holdout(pipeline, holdout, metrics)

    x_holdout = holdout.loc[:, E821EDC_FEATURE_NAMES].to_numpy(dtype=np.float32)
    python_labels = pipeline.predict(x_holdout).astype(np.int64)
    python_probabilities = pipeline.predict_proba(x_holdout).astype(np.float32)
    onnx_labels, onnx_probabilities, input_name, output_names = _run_onnx(
        model_onnx.SerializeToString(), x_holdout
    )
    if not np.array_equal(python_labels, onnx_labels):
        mismatches = int(np.count_nonzero(python_labels != onnx_labels))
        raise ValueError(f"ONNX class parity failed for {mismatches} holdout rows")
    maximum_probability_error = float(
        np.max(np.abs(python_probabilities - onnx_probabilities))
    )
    if maximum_probability_error > 1e-5:
        raise ValueError(
            "ONNX probability parity exceeded 1e-5: "
            f"{maximum_probability_error:.9g}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "e821edc.onnx"
    fixture_path = output_dir / "benchmark-fixtures.json"
    metadata_path = output_dir / "metadata.json"
    onnx_path.write_bytes(model_onnx.SerializeToString())

    selected = select_balanced_fixtures(
        holdout,
        fixtures_per_class=fixtures_per_class,
    )
    fixture_payload = _fixture_payload(
        selected,
        pipeline.predict(selected.loc[:, E821EDC_FEATURE_NAMES].to_numpy(np.float32)),
        pipeline.predict_proba(
            selected.loc[:, E821EDC_FEATURE_NAMES].to_numpy(np.float32)
        ),
    )
    fixture_path.write_text(
        json.dumps(fixture_payload, indent=2) + "\n", encoding="utf-8"
    )

    metadata = {
        "schema_version": 1,
        "run_id": E821EDC_RUN_ID,
        "model_family": "xgboost",
        "window_seconds": 60,
        "step_seconds": 30,
        "feature_names": list(E821EDC_FEATURE_NAMES),
        "labels": {name: value for value, name in LABEL_NAMES.items()},
        "input": {"name": input_name, "shape": [None, 11], "type": "float32"},
        "outputs": output_names,
        "source_model": {
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
        },
        "deployed_model": {
            "sha256": sha256_file(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "mib": onnx_path.stat().st_size / (1024 * 1024),
        },
        "verification": {
            "holdout_rows": len(holdout),
            "class_labels_identical": True,
            "maximum_probability_absolute_error": maximum_probability_error,
            "probability_tolerance": 1e-5,
            "canonical_frame_fingerprint": manifest["frame_fingerprint"],
            "recovered_frame_fingerprint": recovered_fingerprint,
            "canonical_metrics_reproduced": True,
            "fixture_rows": len(selected),
        },
        "imputer": {
            "strategy": pipeline.named_steps["imputer"].strategy,
            "statistics": pipeline.named_steps["imputer"].statistics_.tolist(),
            "indicator_features": pipeline.named_steps[
                "imputer"
            ].indicator_.features_.tolist(),
        },
        "library_versions": _library_versions(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return ExportPaths(onnx_path, metadata_path, fixture_path)


def collector_session_ids_from_log(path: Path) -> list[str]:
    """Recover the immutable collector input membership from an older run log."""
    expected_count: int | None = None
    session_ids: list[str] = []
    seen: set[str] = set()
    scan_pattern = re.compile(r"Collector ingest scan starting: files=(\d+)")
    item_pattern = re.compile(
        r"Collector ingest file \d+/(\d+):.*?/([0-9a-f-]{36})\.json\.gz"
    )
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            scan_match = scan_pattern.search(line)
            if scan_match and expected_count is None:
                expected_count = int(scan_match.group(1))
                continue
            item_match = item_pattern.search(line)
            if not item_match or expected_count is None:
                continue
            if int(item_match.group(1)) != expected_count:
                continue
            session_id = item_match.group(2)
            if session_id not in seen:
                session_ids.append(session_id)
                seen.add(session_id)
            if len(session_ids) == expected_count:
                break
    if expected_count is None:
        raise ValueError(f"Collector scan marker not found in {path}")
    if len(session_ids) != expected_count:
        raise ValueError(
            f"Recovered {len(session_ids)} of {expected_count} collector sessions"
        )
    return session_ids


def select_balanced_fixtures(
    holdout: pd.DataFrame,
    *,
    fixtures_per_class: int,
) -> pd.DataFrame:
    """Choose deterministic, evenly distributed rows from every class."""
    selected: list[pd.DataFrame] = []
    for label in sorted(LABEL_NAMES):
        rows = holdout.loc[holdout["label"].astype(int) == label].sort_values(
            ["session_id", "window_start_ms", "window_end_ms"], kind="stable"
        )
        if len(rows) < fixtures_per_class:
            raise ValueError(
                f"Label {label} has {len(rows)} rows, needs {fixtures_per_class}"
            )
        positions = np.linspace(0, len(rows) - 1, fixtures_per_class, dtype=int)
        selected.append(rows.iloc[positions])
    return pd.concat(selected, ignore_index=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run_metadata(metrics: dict[str, Any]) -> None:
    if tuple(metrics.get("feature_names", ())) != E821EDC_FEATURE_NAMES:
        raise ValueError("Run feature order does not match the e821edc contract")
    if metrics.get("best_params", {}).get("family") != "xgboost":
        raise ValueError("Run e821edc is expected to contain an XGBoost model")
    if metrics.get("collector_holdout", {}).get("rows") != 3941:
        raise ValueError("Run e821edc must contain the 3,941-row holdout")


def _validate_pipeline(pipeline: Any) -> None:
    if list(pipeline.named_steps) != ["imputer", "model"]:
        raise ValueError("Unexpected e821edc pipeline steps")
    if int(pipeline.named_steps["model"].n_features_in_) != len(
        E821EDC_FEATURE_NAMES
    ):
        raise ValueError("Unexpected e821edc model input width")
    if pipeline.named_steps["imputer"].indicator_.features_.size:
        raise ValueError("The selected pipeline unexpectedly adds indicator columns")


def _convert_pipeline(pipeline: Any):
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
        convert_xgboost,
    )
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import (
        calculate_linear_classifier_output_shapes,
    )

    classifier = pipeline.named_steps["model"]
    update_registered_converter(
        type(classifier),
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={
            "nocl": [True, False],
            "zipmap": [True, False, "columns"],
            "raw_scores": [True, False],
        },
    )
    return convert_sklearn(
        pipeline,
        E821EDC_SHORT_RUN_ID,
        [("features", FloatTensorType([None, len(E821EDC_FEATURE_NAMES)]))],
        target_opset={"": 15, "ai.onnx.ml": 3},
        options={id(classifier): {"zipmap": False}},
    )


def _load_run_frame(
    feature_cache: Path,
    collector_session_ids: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    source = _read_feature_dataset(feature_cache / "nor-tmd")
    collector = _read_feature_dataset(feature_cache / "collector")
    membership = set(collector_session_ids)
    collector = collector.loc[
        collector["session_id"].astype(str).isin(membership)
    ].reset_index(drop=True)
    frame = pd.concat([source, collector], ignore_index=True)
    return frame, frame_fingerprint(frame)


def _read_feature_dataset(path: Path) -> pd.DataFrame:
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No feature Parquet parts found: {path}")
    return pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)


def _validate_recovered_holdout(
    pipeline: Any,
    holdout: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    expected = metrics["collector_holdout"]
    if len(holdout) != int(expected["rows"]):
        raise ValueError(
            f"Recovered holdout has {len(holdout)} rows, expected {expected['rows']}"
        )
    x = holdout.loc[:, E821EDC_FEATURE_NAMES].to_numpy(dtype=np.float32)
    observed = confusion_matrix(holdout["label"], pipeline.predict(x)).tolist()
    if observed != expected["confusion_matrix"]:
        raise ValueError("Recovered holdout does not reproduce canonical predictions")


def _run_onnx(
    model_bytes: bytes,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, list[dict[str, Any]]]:
    import onnxruntime as ort

    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    values = session.run(None, {input_name: features})
    outputs = [
        {"name": output.name, "shape": output.shape, "type": output.type}
        for output in session.get_outputs()
    ]
    return (
        np.asarray(values[0], dtype=np.int64),
        np.asarray(values[1], dtype=np.float32),
        input_name,
        outputs,
    )


def _fixture_payload(
    selected: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    cases = []
    for position, (_, row) in enumerate(selected.iterrows()):
        cases.append(
            {
                "id": f"holdout-{position:03d}",
                "session_id": str(row["session_id"]),
                "window_start_ms": int(row["window_start_ms"]),
                "window_end_ms": int(row["window_end_ms"]),
                "features": [float(row[name]) for name in E821EDC_FEATURE_NAMES],
                "expected_label": int(labels[position]),
                "expected_probabilities": [
                    float(value) for value in probabilities[position]
                ],
            }
        )
    return {
        "schema_version": 1,
        "run_id": E821EDC_RUN_ID,
        "feature_names": list(E821EDC_FEATURE_NAMES),
        "labels": {str(value): name for value, name in LABEL_NAMES.items()},
        "cases": cases,
        "feature_extraction_fixture": _feature_extraction_fixture(),
    }


def _feature_extraction_fixture() -> dict[str, Any]:
    readings = [
        {
            "timestamp_ms": index * 50,
            "accelerometer": vector,
            "gyroscope": [vector[2], vector[0], vector[1]],
            "magnetometer": [vector[1] * 10, vector[2] * 10, vector[0] * 10],
        }
        for index, vector in enumerate(
            ([1.0, 2.0, 2.0], [2.0, 3.0, 6.0], [4.0, 0.0, 3.0], [1.0, 2.0, 3.0])
        )
    ]
    vectors = {
        sensor: np.asarray([reading[sensor] for reading in readings], dtype=np.float64)
        for sensor in ("accelerometer", "gyroscope", "magnetometer")
    }
    stats = {
        sensor: aggregate(np.linalg.norm(values, axis=1))
        for sensor, values in vectors.items()
    }
    expected = [
        stats["accelerometer"][name]
        for name in ("mean", "standard_deviation", "range", "minimum")
    ]
    expected.extend(
        stats["gyroscope"][name]
        for name in ("mean", "standard_deviation", "range", "minimum")
    )
    expected.extend(
        stats["magnetometer"][name]
        for name in ("standard_deviation", "range", "minimum")
    )
    return {"readings": readings, "expected_features": expected}


def _library_versions() -> dict[str, str]:
    import onnx
    import onnxmltools
    import onnxruntime
    import sklearn
    import skl2onnx
    import xgboost

    return {
        "onnx": onnx.__version__,
        "onnxmltools": onnxmltools.__version__,
        "onnxruntime": onnxruntime.__version__,
        "scikit_learn": sklearn.__version__,
        "skl2onnx": skl2onnx.__version__,
        "xgboost": xgboost.__version__,
    }
