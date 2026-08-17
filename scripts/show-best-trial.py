"""Show per-mode recall for the best trial in a downloaded AWS run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


SELECTOR_METRICS = (
    "collector_holdout.balanced_accuracy",
    "collector_holdout.accuracy",
    "collector_holdout.macro_f1",
    "collector_holdout.minimum_class_recall",
)
SUMMARY_REPORT_KEYS = {"accuracy", "macro avg", "weighted avg"}


@dataclass(frozen=True)
class TrialResult:
    trial_name: str
    trial_hash: str
    selector_value: float
    recalls: tuple[tuple[str, float], ...]
    metrics_path: Path


def _as_mapping(value: Any, description: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {description} must be a JSON object")
    return value


def _finite_number(value: Any, description: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path}: {description} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {description} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{path}: {description} must be a finite number")
    return number


def _trial_metrics_paths(run_dir: Path) -> list[Path]:
    """Return canonical reports and exclude their MLflow artifact copies."""
    return sorted(run_dir.glob("work/*/reports/*/metrics.json"))


def _read_trial_result(metrics_path: Path, selector_metric: str) -> TrialResult:
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
    metric_name = selector_metric.removeprefix("collector_holdout.")
    selector_value = _finite_number(
        holdout.get(metric_name), selector_metric, metrics_path
    )
    report = _as_mapping(
        holdout.get("classification_report"),
        "collector_holdout.classification_report",
        metrics_path,
    )

    recalls: list[tuple[str, float]] = []
    for mode, mode_report_value in report.items():
        if mode in SUMMARY_REPORT_KEYS:
            continue
        mode_report = _as_mapping(
            mode_report_value,
            f"classification report entry for {mode!r}",
            metrics_path,
        )
        recalls.append(
            (
                mode,
                _finite_number(
                    mode_report.get("recall"),
                    f"collector_holdout.classification_report.{mode}.recall",
                    metrics_path,
                ),
            )
        )
    if not recalls:
        raise ValueError(f"{metrics_path}: classification report has no transport modes")

    trial_hash = metrics.get("trial_hash", metrics.get("config_hash"))
    train_dataset = metrics.get("train_dataset")
    if not isinstance(trial_hash, str) or not trial_hash:
        raise ValueError(
            f"{metrics_path}: trial_hash or config_hash must be a non-empty string"
        )
    if not isinstance(train_dataset, str) or not train_dataset:
        raise ValueError(f"{metrics_path}: train_dataset must be a non-empty string")

    return TrialResult(
        trial_name=f"{train_dataset}-{trial_hash[:8]}",
        trial_hash=trial_hash,
        selector_value=selector_value,
        recalls=tuple(recalls),
        metrics_path=metrics_path,
    )


def find_best_trial(run_dir: Path, selector_metric: str) -> TrialResult:
    metrics_paths = _trial_metrics_paths(run_dir)
    if not metrics_paths:
        raise ValueError(f"no trial metrics.json files found beneath {run_dir}")

    results = [_read_trial_result(path, selector_metric) for path in metrics_paths]
    return min(
        results,
        key=lambda result: (
            -result.selector_value,
            result.trial_name,
            result.trial_hash,
            str(result.metrics_path),
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find the trial with the highest selected collector holdout metric "
            "in a downloaded AWS run and show recall for every transport mode."
        )
    )
    parser.add_argument("selector_metric", choices=SELECTOR_METRICS)
    parser.add_argument("run_id", help="Downloaded run directory name in aws-results")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    results_root: Path | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    if results_root is None:
        results_root = Path(__file__).resolve().parents[1] / "aws-results"

    run_dir = results_root / args.run_id
    if not run_dir.is_dir():
        print(f"error: downloaded run does not exist: {run_dir}", file=sys.stderr)
        return 1

    try:
        result = find_best_trial(run_dir, args.selector_metric)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Best trial in run {args.run_id!r}:")
    print(f"  trial name: {result.trial_name}")
    print(f"  trial hash: {result.trial_hash}")
    print(f"  {args.selector_metric}: {result.selector_value:g}")
    print("  recall by transport mode:")
    for mode, recall in result.recalls:
        print(f"    {mode}: {recall:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
