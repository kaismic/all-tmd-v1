"""Generate a bar chart of the top downloaded MLflow runs by metric."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


METRIC_TITLES = {
    "collector_holdout.accuracy": "Best Overall Accuracy",
    "collector_holdout.macro_f1": "Best Macro F1 Score",
    "collector_holdout.balanced_accuracy": "Best Balanced Accuracy",
}


@dataclass(frozen=True)
class RunMetric:
    run_id: str
    value: float
    metrics_path: Path


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("n must be at least 1")
    return parsed


def output_filename(metric_name: str, n: int) -> str:
    """Return the requested image filename for a metric and result limit."""
    return f"{metric_name}-top-{n}.png"


def _artifact_metrics_paths(results_root: Path) -> list[Path]:
    """Find MLflow artifact metrics, excluding canonical work-report copies."""
    return sorted(
        results_root.glob("*/mlflow/mlartifacts/*/*/artifacts/metrics.json")
    )


def _as_mapping(value: Any, description: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {description} must be a JSON object")
    return value


def _metric_value(value: Any, metric_name: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path}: {metric_name} must be between 0 and 1")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: {metric_name} must be between 0 and 1"
        ) from error
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{path}: {metric_name} must be between 0 and 1")
    return number


def _read_run_metric(metrics_path: Path, metric_name: str) -> RunMetric:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metrics_path}: invalid JSON ({error})") from error

    metrics = _as_mapping(metrics, "metrics", metrics_path)
    section_name, key = metric_name.split(".", maxsplit=1)
    section = _as_mapping(metrics.get(section_name), section_name, metrics_path)
    value = _metric_value(section.get(key), metric_name, metrics_path)
    run_id = metrics_path.parent.parent.name
    if not run_id:
        raise ValueError(f"{metrics_path}: could not determine the MLflow run ID")
    return RunMetric(run_id=run_id, value=value, metrics_path=metrics_path)


def collect_run_metrics(results_root: Path, metric_name: str) -> list[RunMetric]:
    """Read one metric for every unique MLflow run under downloaded results."""
    metrics_paths = _artifact_metrics_paths(results_root)
    if not metrics_paths:
        raise ValueError(
            f"no downloaded MLflow metrics.json files found beneath {results_root}"
        )

    results_by_run_id: dict[str, RunMetric] = {}
    for metrics_path in metrics_paths:
        result = _read_run_metric(metrics_path, metric_name)
        existing = results_by_run_id.get(result.run_id)
        if existing is not None:
            if existing.value != result.value:
                raise ValueError(
                    f"MLflow run {result.run_id!r} has conflicting {metric_name} "
                    f"values in {existing.metrics_path} and {result.metrics_path}"
                )
            continue
        results_by_run_id[result.run_id] = result

    return sorted(
        results_by_run_id.values(),
        key=lambda result: (-result.value, result.run_id),
    )


def build_figure(
    results: Sequence[RunMetric],
    metric_name: str,
):
    """Build the top-run bar chart without requiring an interactive backend."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure_width = max(7.0, 0.8 * len(results) + 2.0)
    figure = Figure(figsize=(figure_width, 5.5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    palette = colormaps["viridis"].resampled(max(1, len(results)))
    bars = axis.bar(
        [result.run_id[:7] for result in results],
        [result.value for result in results],
        color=[palette(index) for index in range(len(results))],
    )
    axis.set_title(METRIC_TITLES[metric_name])
    axis.set_xlabel("Run ID")
    axis.set_ylabel(metric_name)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.bar_label(bars, fmt="%.4f", padding=3)
    figure.tight_layout()
    return figure


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a bar chart of the top downloaded MLflow runs for a "
            "collector-holdout metric."
        )
    )
    parser.add_argument("metric_name", choices=tuple(METRIC_TITLES))
    parser.add_argument("n", type=_positive_integer, help="maximum runs to chart")
    parser.add_argument(
        "--results-root",
        type=Path,
        help="downloaded results root (default: repository aws-results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "image destination (default: <results-root>/top-metric-charts)"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    results_root: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    if results_root is None:
        results_root = (
            args.results_root
            if args.results_root is not None
            else Path(__file__).resolve().parents[1] / "aws-results"
        )
    elif args.results_root is not None:
        raise ValueError("results_root and --results-root cannot both be supplied")

    if output_dir is None:
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else results_root / "top-metric-charts"
        )
    elif args.output_dir is not None:
        raise ValueError("output_dir and --output-dir cannot both be supplied")

    if not results_root.is_dir():
        print(f"error: results directory does not exist: {results_root}", file=sys.stderr)
        return 1

    try:
        results = collect_run_metrics(results_root, args.metric_name)[: args.n]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename(args.metric_name, args.n)
        figure = build_figure(results, args.metric_name)
        try:
            figure.savefig(output_path, dpi=150)
        finally:
            figure.clear()
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
