"""Export top collector-holdout metric tables across downloaded AWS runs."""

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
class RankingMetric:
    key: str
    caption: str
    column_heading: str


RANKING_METRICS = (
    RankingMetric("macro_f1", "Best F1 Score", "average"),
    RankingMetric("accuracy", "Best Accuracy", "accuracy"),
    RankingMetric(
        "balanced_accuracy",
        "Best Balanced Accuracy",
        "balanced accuracy",
    ),
)


@dataclass(frozen=True)
class TrialScores:
    run_id: str
    ranking_values: dict[str, float]
    mode_f1: dict[str, float]
    metrics_path: Path


def _as_mapping(value: Any, description: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {description} must be a JSON object")
    return value


def _score(value: Any, description: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path}: {description} must be between 0 and 1")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {description} must be between 0 and 1") from error
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{path}: {description} must be between 0 and 1")
    return number


def _artifact_metrics_paths(results_root: Path) -> list[Path]:
    return sorted(
        results_root.glob("*/mlflow/mlartifacts/*/*/artifacts/metrics.json")
    )


def _read_trial_scores(metrics_path: Path) -> TrialScores:
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
    ranking_values = {
        metric.key: _score(
            holdout.get(metric.key),
            f"collector_holdout.{metric.key}",
            metrics_path,
        )
        for metric in RANKING_METRICS
    }
    report = _as_mapping(
        holdout.get("classification_report"),
        "collector_holdout.classification_report",
        metrics_path,
    )

    mode_f1: dict[str, float] = {}
    for mode, mode_report_value in report.items():
        if mode in SUMMARY_REPORT_KEYS:
            continue
        mode_report = _as_mapping(
            mode_report_value,
            f"classification report entry for {mode!r}",
            metrics_path,
        )
        mode_f1[mode] = _score(
            mode_report.get("f1-score"),
            f"collector_holdout.classification_report.{mode}.f1-score",
            metrics_path,
        )
    if not mode_f1:
        raise ValueError(f"{metrics_path}: classification report has no transport modes")

    run_id = metrics_path.parent.parent.name
    if not run_id:
        raise ValueError(f"{metrics_path}: could not determine the MLflow run ID")
    return TrialScores(
        run_id=run_id,
        ranking_values=ranking_values,
        mode_f1=mode_f1,
        metrics_path=metrics_path,
    )


def collect_trials(results_root: Path) -> list[TrialScores]:
    metrics_paths = _artifact_metrics_paths(results_root)
    if not metrics_paths:
        raise ValueError(
            f"no downloaded MLflow metrics.json files found beneath {results_root}"
        )

    trials_by_run_id: dict[str, TrialScores] = {}
    for metrics_path in metrics_paths:
        trial = _read_trial_scores(metrics_path)
        existing = trials_by_run_id.get(trial.run_id)
        if existing is not None:
            if (existing.ranking_values, existing.mode_f1) != (
                trial.ranking_values,
                trial.mode_f1,
            ):
                raise ValueError(
                    f"MLflow run {trial.run_id!r} has conflicting metrics in "
                    f"{existing.metrics_path} and {trial.metrics_path}"
                )
            continue
        trials_by_run_id[trial.run_id] = trial

    return list(trials_by_run_id.values())


def rank_trials(
    trials: Sequence[TrialScores],
    metric: RankingMetric,
    limit: int,
) -> list[TrialScores]:
    return sorted(
        trials,
        key=lambda trial: (-trial.ranking_values[metric.key], trial.run_id),
    )[:limit]


def _latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def render_latex_table(
    trials: Sequence[TrialScores],
    metric: RankingMetric,
    *,
    precision: int,
) -> str:
    modes = sorted({mode for trial in trials for mode in trial.mode_f1})
    columns = "c|" + "c" * len(modes) + "|c"
    lines = [
        rf"\caption{{{metric.caption}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"    \hline",
        "    Run ID & "
        + " & ".join(_latex_escape(mode) for mode in modes)
        + f" & {_latex_escape(metric.column_heading)} "
        + r"\\",
        r"    \hline",
    ]
    for trial in trials:
        values = [
            f"{trial.mode_f1[mode]:.{precision}f}"
            if mode in trial.mode_f1
            else "--"
            for mode in modes
        ]
        lines.append(
            "    "
            + _latex_escape(trial.run_id)
            + " & "
            + " & ".join(values)
            + f" & {trial.ranking_values[metric.key]:.{precision}f} \\\\"
        )
    lines.extend((r"    \hline", r"\end{tabular}"))
    return "\n".join(lines)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("precision must be at least 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print LaTeX tables for the trials with the highest collector-holdout "
            "macro F1, accuracy, and balanced accuracy across every downloaded "
            "AWS run."
        )
    )
    parser.add_argument(
        "--limit",
        type=_positive_integer,
        default=3,
        help="number of trials to display (default: 3)",
    )
    parser.add_argument(
        "--precision",
        type=_non_negative_integer,
        default=4,
        help="number of digits after the decimal point (default: 4)",
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

    try:
        trials = collect_trials(results_root)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    tables = [
        render_latex_table(
            rank_trials(trials, metric, args.limit),
            metric,
            precision=args.precision,
        )
        for metric in RANKING_METRICS
    ]
    print("\n\n".join(tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
