"""Export per-transport-mode recall for every trial in an AWS run as LaTeX."""

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
class TrialRecall:
    trial_name: str
    trial_hash: str
    trial_index: int | None
    recalls: dict[str, float]
    metrics_path: Path


def _as_mapping(value: Any, description: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {description} must be a JSON object")
    return value


def _recall(value: Any, mode: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path}: recall for {mode!r} must be between 0 and 1")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: recall for {mode!r} must be between 0 and 1"
        ) from error
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{path}: recall for {mode!r} must be between 0 and 1")
    return number


def _trial_metrics_paths(run_dir: Path) -> list[Path]:
    """Return canonical reports and exclude their MLflow artifact copies."""
    return sorted(run_dir.glob("work/*/reports/*/metrics.json"))


def _read_trial_recall(metrics_path: Path) -> TrialRecall:
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

    recalls: dict[str, float] = {}
    for mode, mode_report_value in report.items():
        if mode in SUMMARY_REPORT_KEYS:
            continue
        mode_report = _as_mapping(
            mode_report_value,
            f"classification report entry for {mode!r}",
            metrics_path,
        )
        recalls[mode] = _recall(mode_report.get("recall"), mode, metrics_path)
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

    trial_index = metrics.get("trial_index")
    if trial_index is not None and (
        isinstance(trial_index, bool) or not isinstance(trial_index, int)
    ):
        raise ValueError(f"{metrics_path}: trial_index must be an integer or null")

    return TrialRecall(
        trial_name=trial_name,
        trial_hash=trial_hash,
        trial_index=trial_index,
        recalls=recalls,
        metrics_path=metrics_path,
    )


def collect_trial_recalls(run_dir: Path) -> tuple[list[TrialRecall], list[str]]:
    metrics_paths = _trial_metrics_paths(run_dir)
    if not metrics_paths:
        raise ValueError(f"no trial metrics.json files found beneath {run_dir}")

    trials = [_read_trial_recall(path) for path in metrics_paths]
    trials.sort(
        key=lambda trial: (
            trial.trial_index is None,
            trial.trial_index if trial.trial_index is not None else 0,
            trial.trial_name,
            trial.trial_hash,
            str(trial.metrics_path),
        )
    )
    modes = sorted({mode for trial in trials for mode in trial.recalls})
    return trials, modes


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
    trials: Sequence[TrialRecall],
    modes: Sequence[str],
    *,
    precision: int,
) -> str:
    columns = "l" + "r" * len(modes)
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\hline",
        "Run & " + " & ".join(_latex_escape(mode) for mode in modes) + r" \\",
        r"\hline",
    ]
    for trial in trials:
        values = [
            f"{trial.recalls[mode]:.{precision}f}" if mode in trial.recalls else "--"
            for mode in modes
        ]
        lines.append(
            _latex_escape(trial.trial_name) + " & " + " & ".join(values) + r" \\"
        )
    lines.extend((r"\hline", r"\end{tabular}"))
    return "\n".join(lines)


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("precision must be at least 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print a LaTeX table containing collector-holdout recall for every "
            "trial and transport mode in a downloaded AWS run."
        )
    )
    parser.add_argument("run_id", help="Downloaded run directory name in aws-results")
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

    run_dir = results_root / args.run_id
    if not run_dir.is_dir():
        print(f"error: downloaded run does not exist: {run_dir}", file=sys.stderr)
        return 1

    try:
        trials, modes = collect_trial_recalls(run_dir)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(render_latex_table(trials, modes, precision=args.precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
