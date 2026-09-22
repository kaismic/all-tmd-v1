"""Generate tables and figures for one downloaded AWS experiment run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable, Sequence


EXPERIMENT_NAME = "ALL-TMD"
SUMMARY_REPORT_KEYS = {"accuracy", "macro avg", "weighted avg"}
TABLES_FILENAME = "aws-run-tables.tex"
WINDOW_FIGURE = "average-collector-holdout-macro-f1-by-window-seconds.png"


@dataclass(frozen=True)
class TrialResult:
    run_uuid: str
    trial_index: int
    params: dict[str, str]
    metrics: dict[str, Any]

    @property
    def window_seconds(self) -> int:
        return int(self.params["window_seconds"])

    @property
    def step_seconds(self) -> int:
        return int(self.params["step_seconds"])

    @property
    def holdout(self) -> dict[str, Any]:
        return self.metrics["collector_holdout"]

    @property
    def macro_f1(self) -> float:
        return float(self.holdout["macro_f1"])

    @property
    def balanced_accuracy(self) -> float:
        return float(self.holdout["balanced_accuracy"])

    @property
    def accuracy(self) -> float:
        return float(self.holdout["accuracy"])

    @property
    def cross_validation_macro_f1(self) -> float:
        return float(self.metrics["best_cross_validation_macro_f1"])

    @property
    def class_f1(self) -> dict[str, float]:
        report = self.holdout["classification_report"]
        return {
            label: float(values["f1-score"])
            for label, values in report.items()
            if label not in SUMMARY_REPORT_KEYS
        }


@dataclass(frozen=True)
class Average:
    label: int | str
    macro_f1: float
    run_count: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _timestamp_milliseconds(value: Any, description: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"run summary {description} must be an ISO timestamp")
    try:
        return int(datetime.fromisoformat(value).timestamp() * 1000)
    except ValueError as error:
        raise ValueError(f"run summary {description} is invalid: {value!r}") from error


def _validate_score(value: Any, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be between 0 and 1")
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be between 0 and 1") from error
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{description} must be between 0 and 1")
    return score


def _metrics_path(run_dir: Path, run_uuid: str) -> Path:
    matches = sorted(
        run_dir.glob(f"mlflow/mlartifacts/*/{run_uuid}/artifacts/metrics.json")
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one metrics artifact for MLflow run {run_uuid}, found {len(matches)}"
        )
    return matches[0]


def load_trials(run_dir: Path, summary: dict[str, Any]) -> list[TrialResult]:
    """Load only the MLflow runs started during the downloaded AWS run."""
    database_path = run_dir / "mlflow" / "mlflow.db"
    if not database_path.is_file():
        raise ValueError(f"MLflow database does not exist: {database_path}")

    started_at = _timestamp_milliseconds(summary.get("started_at"), "started_at")
    completed_at = _timestamp_milliseconds(summary.get("completed_at"), "completed_at")
    query = """
        SELECT r.run_uuid
        FROM experiments AS experiment
        JOIN runs AS r ON r.experiment_id = experiment.experiment_id
        WHERE experiment.name = ?
          AND experiment.lifecycle_stage = 'active'
          AND r.lifecycle_stage = 'active'
          AND r.status = 'FINISHED'
          AND r.start_time BETWEEN ? AND ?
        ORDER BY r.start_time, r.run_uuid
    """
    try:
        with sqlite3.connect(database_path) as connection:
            run_uuids = [
                str(row[0])
                for row in connection.execute(
                    query, (EXPERIMENT_NAME, started_at, completed_at)
                ).fetchall()
            ]
            if not run_uuids:
                raise ValueError("no finished MLflow runs fall inside the AWS run interval")
            placeholders = ",".join("?" for _ in run_uuids)
            parameter_rows = connection.execute(
                f"SELECT run_uuid, key, value FROM params "
                f"WHERE run_uuid IN ({placeholders})",
                run_uuids,
            ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"could not read MLflow database: {error}") from error

    params_by_run: dict[str, dict[str, str]] = defaultdict(dict)
    for run_uuid, key, value in parameter_rows:
        params_by_run[str(run_uuid)][str(key)] = str(value)

    trials: list[TrialResult] = []
    for run_uuid in run_uuids:
        params = params_by_run[run_uuid]
        for required in ("trial_index", "window_seconds", "step_seconds"):
            if required not in params:
                raise ValueError(f"MLflow run {run_uuid} is missing parameter {required!r}")
        metrics = _read_json(_metrics_path(run_dir, run_uuid))
        holdout = metrics.get("collector_holdout")
        if not isinstance(holdout, dict):
            raise ValueError(f"MLflow run {run_uuid} has no collector_holdout metrics")
        for key in ("macro_f1", "balanced_accuracy", "accuracy"):
            _validate_score(holdout.get(key), f"{run_uuid} collector_holdout.{key}")
        _validate_score(
            metrics.get("best_cross_validation_macro_f1"),
            f"{run_uuid} best_cross_validation_macro_f1",
        )
        trials.append(
            TrialResult(
                run_uuid=run_uuid,
                trial_index=int(params["trial_index"]),
                params=params,
                metrics=metrics,
            )
        )

    expected_count = int(summary.get("trial_count", -1))
    if len(trials) != expected_count:
        raise ValueError(
            f"run summary expects {expected_count} trials, but {len(trials)} were found"
        )
    indices = [trial.trial_index for trial in trials]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(f"trial indices are incomplete or duplicated: {indices}")
    return sorted(trials, key=lambda trial: trial.trial_index)


def report_kind(trials: Sequence[TrialResult]) -> str:
    if any(
        key.startswith("calibration_fraction.")
        for trial in trials
        for key in trial.params
    ):
        return "calibration"
    pressure_values = {trial.params.get("features.pressure", "") for trial in trials}
    if len(pressure_values) > 1:
        return "pressure"
    magnetometer_values = {
        trial.params.get("features.magnetometer", "") for trial in trials
    }
    if len(magnetometer_values) > 1:
        return "magnetometer"
    raise ValueError("could not identify a varying experimental factor")


def _feature_label(value: str) -> str:
    names = {
        "standard_deviation": "Std. dev.",
        "range": "range",
        "minimum": "minimum",
        "mean": "mean",
        "delta_from_window_start": "window-start delta",
        "delta_from_session_baseline": "session-baseline delta",
    }
    return " + ".join(names.get(item, item.replace("_", " ")) for item in value.split(",") if item)


def variant_label(trial: TrialResult, kind: str) -> str:
    if kind == "pressure":
        value = trial.params.get("features.pressure", "")
        return _feature_label(value) if value else "No pressure"
    if kind == "magnetometer":
        return _feature_label(trial.params.get("features.magnetometer", ""))
    if kind == "calibration":
        allocations = sorted(
            (
                key.removeprefix("calibration_fraction."),
                float(value),
            )
            for key, value in trial.params.items()
            if key.startswith("calibration_fraction.")
        )
        return ", ".join(
            f"{label.title()} {fraction:.0%}" for label, fraction in allocations
        )
    raise ValueError(f"unsupported report kind: {kind}")


def short_variant_label(trial: TrialResult, kind: str) -> str:
    if kind != "calibration":
        return variant_label(trial, kind).replace(" + ", " +\n")
    allocations = [
        (key.removeprefix("calibration_fraction."), float(value))
        for key, value in trial.params.items()
        if key.startswith("calibration_fraction.")
    ]
    highest = max(fraction for _, fraction in allocations)
    high_labels = [label.title() for label, fraction in allocations if fraction == highest]
    return f"{'/'.join(high_labels)} {highest:.0%}"


def chart_variant_label(variant: str, kind: str) -> str:
    """Return a compact factor label that stays readable on a bar chart."""
    if kind == "pressure":
        if variant == "No pressure":
            return variant
        return (
            variant.replace("Std. dev. + range + ", "")
            .replace("window-start", "Window-start")
            .replace("session-baseline", "Session-baseline")
        )
    if kind == "magnetometer":
        return variant.replace("Std. dev. + range + ", "+ ")
    return variant


def ordered_variants(trials: Sequence[TrialResult], kind: str) -> list[str]:
    return list(dict.fromkeys(variant_label(trial, kind) for trial in trials))


def averages_for(
    trials: Iterable[TrialResult],
    labels: Sequence[int | str],
    value_for,
) -> list[Average]:
    grouped: dict[int | str, list[float]] = defaultdict(list)
    for trial in trials:
        grouped[value_for(trial)].append(trial.macro_f1)
    return [
        Average(label, sum(grouped[label]) / len(grouped[label]), len(grouped[label]))
        for label in labels
    ]


def latex_escape(value: str) -> str:
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


def _table(caption: str, label: str, headings: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    alignment = "l" + "r" * (len(headings) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(f"\\textbf{{{latex_escape(heading)}}}" for heading in headings) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return lines


def _unique_param(trials: Sequence[TrialResult], key: str, default: str = "--") -> str:
    values = sorted({trial.params[key] for trial in trials if key in trial.params})
    return values[0] if len(values) == 1 else default


def _parameter_summary(trials: Sequence[TrialResult], key: str) -> str:
    values = sorted({trial.params[key] for trial in trials if key in trial.params})
    if not values:
        return "--"
    if len(values) == 1:
        return values[0]
    try:
        numbers = sorted(int(value) for value in values)
    except ValueError:
        return ", ".join(values)
    return f"{numbers[0]}--{numbers[-1]} (configuration-dependent)"


def _format_elapsed(seconds: Any) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _summary_rows(
    run_id: str,
    summary: dict[str, Any],
    trials: Sequence[TrialResult],
    kind: str,
) -> list[list[str]]:
    labels = sorted({label for trial in trials for label in trial.class_f1})
    windows = sorted({(trial.window_seconds, trial.step_seconds) for trial in trials})
    factor_names = {
        "pressure": "Pressure-feature ablation",
        "magnetometer": "Magnetometer-feature ablation",
        "calibration": "Class-specific calibration allocation",
    }
    return [
        ["Run ID", latex_escape(run_id)],
        ["Git commit", latex_escape(str(summary.get("git_commit", "--")))],
        ["Experiment", latex_escape(factor_names[kind])],
        ["Trials", str(len(trials))],
        ["Training dataset", latex_escape(_unique_param(trials, "train_dataset"))],
        ["Transport modes", latex_escape(", ".join(label.title() for label in labels))],
        [
            "Effective collector sessions",
            latex_escape(_parameter_summary(trials, "collector_session_count")),
        ],
        [
            "Window / step pairs",
            latex_escape(", ".join(f"{window}s / {step}s" for window, step in windows)),
        ],
        ["Optuna trials per configuration", latex_escape(_unique_param(trials, "optuna_trials"))],
        ["Candidate model families", latex_escape(_unique_param(trials, "model_families"))],
        ["Elapsed time", _format_elapsed(summary.get("duration_seconds", 0))],
    ]


def render_factorial_latex(
    run_id: str,
    summary: dict[str, Any],
    trials: Sequence[TrialResult],
    kind: str,
) -> str:
    windows = sorted({trial.window_seconds for trial in trials})
    variants = ordered_variants(trials, kind)
    lookup = {(trial.window_seconds, variant_label(trial, kind)): trial for trial in trials}
    expected = {(window, variant) for window in windows for variant in variants}
    if set(lookup) != expected:
        raise ValueError("trials do not form a complete window/factor grid")
    window_averages = averages_for(trials, windows, lambda trial: trial.window_seconds)
    variant_averages = averages_for(trials, variants, lambda trial: variant_label(trial, kind))
    factor_caption = "pressure configuration" if kind == "pressure" else "magnetometer features"
    label_prefix = "aws-run-" + run_id.split("-")[1]

    lines = [
        f"% Generated by generate-aws-run-exploratory-report.py for {run_id}.",
        "% Macro F1 values are arithmetic means across the indicated finished trials.",
    ]
    lines.extend(
        _table(
            "Average collector holdout macro F1 by window duration.",
            f"tab:{label_prefix}-window-f1",
            ("Window (seconds)", "Average Macro F1", "Trials"),
            [
                (str(item.label), f"{item.macro_f1:.4f}", str(item.run_count))
                for item in window_averages
            ],
        )
    )
    lines.extend(
        _table(
            f"Average collector holdout macro F1 by {factor_caption}.",
            f"tab:{label_prefix}-factor-f1",
            (factor_caption.title(), "Average Macro F1", "Trials"),
            [
                (latex_escape(str(item.label)), f"{item.macro_f1:.4f}", str(item.run_count))
                for item in variant_averages
            ],
        )
    )
    matrix_rows = []
    for window in windows:
        matrix_rows.append(
            [str(window)]
            + [f"{lookup[(window, variant)].macro_f1:.4f}" for variant in variants]
        )
    lines.extend(
        _table(
            f"Collector holdout macro F1 for each window and {factor_caption} combination.",
            f"tab:{label_prefix}-trial-f1",
            ("Window (seconds)", *variants),
            matrix_rows,
        )
    )
    lines.extend(
        _table(
            "AWS experiment run summary.",
            f"tab:{label_prefix}-summary",
            ("Metric", "Value"),
            _summary_rows(run_id, summary, trials, kind),
        )
    )
    return "\n".join(lines)


def render_calibration_latex(
    run_id: str,
    summary: dict[str, Any],
    trials: Sequence[TrialResult],
) -> str:
    label_prefix = "aws-run-" + run_id.split("-")[1]
    labels = sorted({label for trial in trials for label in trial.class_f1})
    lines = [
        f"% Generated by generate-aws-run-exploratory-report.py for {run_id}.",
        "% Calibration allocations are fractions of each mode assigned to calibration.",
    ]
    lines.extend(
        _table(
            "Performance by class-specific collector calibration allocation.",
            f"tab:{label_prefix}-allocation-metrics",
            (
                "Calibration allocation",
                "CV Macro F1",
                "Holdout Macro F1",
                "Balanced Accuracy",
                "Accuracy",
            ),
            [
                (
                    latex_escape(variant_label(trial, "calibration")),
                    f"{trial.cross_validation_macro_f1:.4f}",
                    f"{trial.macro_f1:.4f}",
                    f"{trial.balanced_accuracy:.4f}",
                    f"{trial.accuracy:.4f}",
                )
                for trial in trials
            ],
        )
    )
    lines.extend(
        _table(
            "Per-mode collector holdout F1 by calibration allocation.",
            f"tab:{label_prefix}-allocation-class-f1",
            ("Calibration allocation", *(label.title() for label in labels)),
            [
                (latex_escape(variant_label(trial, "calibration")),)
                + tuple(f"{trial.class_f1[label]:.4f}" for label in labels)
                for trial in trials
            ],
        )
    )
    lines.extend(
        _table(
            "AWS experiment run summary.",
            f"tab:{label_prefix}-summary",
            ("Metric", "Value"),
            _summary_rows(run_id, summary, trials, "calibration"),
        )
    )
    return "\n".join(lines)


def plot_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(max(8.2, len(labels) * 1.5 + 2.5), 5.5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    palette = colormaps["viridis"].resampled(len(labels))
    bars = axis.bar(labels, values, color=[palette(index) for index in range(len(labels))], width=0.68)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Average Macro F1")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.bar_label(bars, fmt="%.4f", padding=4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    figure.clear()


def plot_grouped_chart(
    category_labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9.4, 5.8))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = np.arange(len(category_labels), dtype=float)
    width = 0.78 / len(series)
    palette = colormaps["viridis"].resampled(len(series))
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        bars = axis.bar(positions + offset, values, width, label=name, color=palette(index))
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(positions, category_labels)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    figure.clear()


def render_readme(run_id: str, kind: str, filenames: Sequence[str]) -> str:
    descriptions = {
        "pressure": "pressure-feature and window-duration ablation",
        "magnetometer": "magnetometer-feature and window-duration ablation",
        "calibration": "class-specific collector calibration-allocation comparison",
    }
    file_lines = "\n".join(f"- `{filename}`" for filename in filenames)
    return f"""# Exploratory report for `{run_id}`

This directory contains the generated tables and figures for the run's
{descriptions[kind]}.

{file_lines}

Regenerate the report from the repository root with:

```powershell
python .\\scripts\\generate-aws-run-exploratory-report.py {run_id}
```
"""


def generate_report(run_dir: Path, output_dir: Path) -> list[Path]:
    summary_path = run_dir / "run" / "run-summary.json"
    if not summary_path.is_file():
        raise ValueError(f"run summary does not exist: {summary_path}")
    summary = _read_json(summary_path)
    run_id = str(summary.get("run_id") or run_dir.name)
    if run_id != run_dir.name:
        raise ValueError(f"run summary ID {run_id!r} does not match {run_dir.name!r}")
    trials = load_trials(run_dir, summary)
    kind = report_kind(trials)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    tables_path = output_dir / TABLES_FILENAME
    if kind == "calibration":
        tables_path.write_text(
            render_calibration_latex(run_id, summary, trials), encoding="utf-8"
        )
        overall_figure = output_dir / "collector-holdout-overall-metrics-by-calibration-allocation.png"
        class_figure = output_dir / "collector-holdout-class-f1-by-calibration-allocation.png"
        chart_labels = [short_variant_label(trial, kind) for trial in trials]
        plot_grouped_chart(
            chart_labels,
            (
                ("Macro F1", [trial.macro_f1 for trial in trials]),
                ("Balanced accuracy", [trial.balanced_accuracy for trial in trials]),
                ("Accuracy", [trial.accuracy for trial in trials]),
            ),
            "Collector Holdout Metrics by Calibration Allocation",
            "Score",
            overall_figure,
        )
        class_labels = sorted({label for trial in trials for label in trial.class_f1})
        plot_grouped_chart(
            chart_labels,
            tuple(
                (label.title(), [trial.class_f1[label] for trial in trials])
                for label in class_labels
            ),
            "Per-Mode Collector Holdout F1 by Calibration Allocation",
            "F1 Score",
            class_figure,
        )
        output_paths.extend((tables_path, overall_figure, class_figure))
    else:
        tables_path.write_text(
            render_factorial_latex(run_id, summary, trials, kind), encoding="utf-8"
        )
        windows = sorted({trial.window_seconds for trial in trials})
        variants = ordered_variants(trials, kind)
        window_averages = averages_for(trials, windows, lambda trial: trial.window_seconds)
        variant_averages = averages_for(
            trials, variants, lambda trial: variant_label(trial, kind)
        )
        factor_filename = (
            "average-collector-holdout-macro-f1-by-pressure-configuration.png"
            if kind == "pressure"
            else "average-collector-holdout-macro-f1-by-magnetometer-features.png"
        )
        window_figure = output_dir / WINDOW_FIGURE
        factor_figure = output_dir / factor_filename
        plot_bar_chart(
            [str(item.label) for item in window_averages],
            [item.macro_f1 for item in window_averages],
            "Average Collector Holdout Macro F1 by Window Duration",
            "Window Seconds",
            window_figure,
        )
        plot_bar_chart(
            [chart_variant_label(str(item.label), kind) for item in variant_averages],
            [item.macro_f1 for item in variant_averages],
            "Average Collector Holdout Macro F1 by "
            + ("Pressure Configuration" if kind == "pressure" else "Magnetometer Features"),
            "Pressure Configuration" if kind == "pressure" else "Magnetometer Features",
            factor_figure,
        )
        output_paths.extend((tables_path, window_figure, factor_figure))

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        render_readme(run_id, kind, [path.name for path in output_paths]),
        encoding="utf-8",
    )
    return [readme_path, *output_paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="downloaded AWS RunId")
    parser.add_argument(
        "--results-root",
        type=Path,
        help="downloaded results root (default: repository aws-results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="report destination (default: <run-dir>/exploratory-report)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(__file__).resolve().parents[1]
    results_root = args.results_root or project_dir / "aws-results"
    run_dir = results_root / args.run_id
    if not run_dir.is_dir():
        print(f"error: downloaded run does not exist: {run_dir}", file=sys.stderr)
        return 1
    output_dir = args.output_dir or run_dir / "exploratory-report"
    try:
        paths = generate_report(run_dir, output_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
