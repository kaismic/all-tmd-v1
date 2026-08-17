"""Rank downloaded AWS trials by per-label collector holdout accuracy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


SUMMARY_REPORT_KEYS = {"accuracy", "macro avg", "weighted avg"}


@dataclass(frozen=True)
class TrialResult:
    trial_name: str
    trial_hash: str
    predicted_labels: tuple[str, ...]
    normalized_predicted_values: tuple[float, ...]
    support: float
    normalized_true_label_accuracy: float
    metrics_path: Path


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("top_n must be at least 1")
    return parsed


def _trial_metrics_paths(results_root: Path) -> list[Path]:
    """Return trial metrics while excluding duplicate MLflow artifact copies."""
    return sorted(
        path
        for path in results_root.rglob("metrics.json")
        if len(path.parents) >= 4
        and path.parents[1].name == "reports"
        and path.parents[3].name == "work"
    )


def _as_mapping(value: Any, description: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {description} must be a JSON object")
    return value


def _read_trial_result(
    metrics_path: Path,
    transport_mode: str,
) -> tuple[TrialResult | None, tuple[str, ...]]:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metrics_path}: invalid JSON ({error})") from error

    metrics = _as_mapping(metrics, "metrics", metrics_path)
    holdout = _as_mapping(
        metrics.get("collector_holdout"),
        "collector_holdout",
        metrics_path,
    )
    report = _as_mapping(
        holdout.get("classification_report"),
        "collector_holdout.classification_report",
        metrics_path,
    )
    predicted_labels = tuple(
        key for key in report if key not in SUMMARY_REPORT_KEYS
    )
    if not predicted_labels:
        raise ValueError(f"{metrics_path}: classification report has no labels")
    if transport_mode not in predicted_labels:
        return None, predicted_labels

    matrix = holdout.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(predicted_labels)
        or any(not isinstance(row, list) or len(row) != len(predicted_labels) for row in matrix)
    ):
        raise ValueError(
            f"{metrics_path}: collector_holdout.confusion_matrix dimensions "
            "do not match the classification report labels"
        )

    label_index = predicted_labels.index(transport_mode)
    try:
        row = tuple(float(value) for value in matrix[label_index])
        label_report = _as_mapping(
            report[transport_mode],
            f"classification report entry for {transport_mode!r}",
            metrics_path,
        )
        support = float(label_report["support"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{metrics_path}: confusion values and support must be numeric"
        ) from error

    if support < 0 or not math.isfinite(support) or any(
        value < 0 or not math.isfinite(value) for value in row
    ):
        raise ValueError(
            f"{metrics_path}: confusion values and support must be finite and non-negative"
        )
    row_total = sum(row)
    if not math.isclose(row_total, support, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"{metrics_path}: {transport_mode!r} support ({support:g}) does not "
            f"match its confusion-matrix row total ({row_total:g})"
        )

    normalized = (
        tuple(value / support for value in row)
        if support
        else tuple(0.0 for _ in row)
    )
    trial_hash = metrics.get("trial_hash", metrics.get("config_hash"))
    train_dataset = metrics.get("train_dataset")
    if not isinstance(trial_hash, str) or not trial_hash:
        raise ValueError(
            f"{metrics_path}: trial_hash or config_hash must be a non-empty string"
        )
    if not isinstance(train_dataset, str) or not train_dataset:
        raise ValueError(f"{metrics_path}: train_dataset must be a non-empty string")
    configured_run_name = metrics.get("run_name")
    if configured_run_name is not None and (
        not isinstance(configured_run_name, str) or not configured_run_name.strip()
    ):
        raise ValueError(f"{metrics_path}: run_name must be a non-empty string or null")
    generated_name = f"{train_dataset}-{trial_hash[:8]}"
    trial_name = (
        generated_name
        if configured_run_name is None
        else f"{configured_run_name.strip()}-{generated_name}"
    )

    return (
        TrialResult(
            trial_name=trial_name,
            trial_hash=trial_hash,
            predicted_labels=predicted_labels,
            normalized_predicted_values=normalized,
            support=support,
            normalized_true_label_accuracy=normalized[label_index],
            metrics_path=metrics_path,
        ),
        predicted_labels,
    )


def collect_trial_results(
    results_root: Path,
    transport_mode: str,
) -> tuple[list[TrialResult], set[str], int]:
    metrics_paths = _trial_metrics_paths(results_root)
    results: list[TrialResult] = []
    available_modes: set[str] = set()
    for metrics_path in metrics_paths:
        result, labels = _read_trial_result(metrics_path, transport_mode)
        available_modes.update(labels)
        if result is not None:
            results.append(result)

    results.sort(
        key=lambda result: (
            -result.normalized_true_label_accuracy,
            result.trial_name,
            result.trial_hash,
            str(result.metrics_path),
        )
    )
    return results, available_modes, len(metrics_paths)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank downloaded AWS trials by normalized true-label accuracy in "
            "their collector holdout confusion matrices."
        )
    )
    parser.add_argument("transport_mode", help="True transport label to compare")
    parser.add_argument(
        "top_n",
        type=_positive_integer,
        help="Maximum number of ranked trials to display",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    results_root: Path | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    if results_root is None:
        results_root = Path(__file__).resolve().parents[1] / "aws-results"

    if not results_root.is_dir():
        print(f"error: results directory does not exist: {results_root}", file=sys.stderr)
        return 1

    try:
        results, available_modes, metrics_count = collect_trial_results(
            results_root,
            args.transport_mode,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if metrics_count == 0:
        print(
            f"error: no trial metrics.json files found beneath {results_root}",
            file=sys.stderr,
        )
        return 1
    if not results:
        modes = ", ".join(sorted(available_modes)) or "none"
        print(
            f"error: transport mode {args.transport_mode!r} was not found; "
            f"available modes: {modes}",
            file=sys.stderr,
        )
        return 1

    displayed = results[: args.top_n]
    print(
        f"Top {len(displayed)} of {len(results)} trials for true label "
        f"{args.transport_mode!r} (collector holdout):"
    )
    for rank, result in enumerate(displayed, start=1):
        label_order = ", ".join(result.predicted_labels)
        values = json.dumps(result.normalized_predicted_values)
        print(f"{rank}. trial name: {result.trial_name}")
        print(f"   trial hash: {result.trial_hash}")
        print(f"   normalized true-label accuracy: {result.normalized_true_label_accuracy:g}")
        print(f"   normalized predicted values [{label_order}]: {values}")
        print(f"   support: {result.support:g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
