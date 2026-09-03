"""Plot average collector-holdout macro F1 by MLflow window size."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import sys
from typing import Sequence


DEFAULT_DATABASE = Path(r"D:\tmd-data\all-tmd-work\mlflow.db")
DEFAULT_OUTPUT_DIR = Path(r"D:\tmd-data\all-tmd-work\model-metric-figures")
DEFAULT_EXPERIMENT = "ALL-TMD Transfer v2"
OUTPUT_FILENAME = "average-collector-holdout-macro-f1-by-window-seconds.png"
WINDOW_PARAMETER = "window_seconds"
F1_METRIC = "collector_holdout.macro_f1"


@dataclass(frozen=True)
class WindowF1Average:
    window_seconds: float
    average_f1: float
    run_count: int


def load_window_f1_averages(
    database_path: Path,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> list[WindowF1Average]:
    """Load active MLflow runs and average their latest F1 by window size."""
    if not database_path.is_file():
        raise ValueError(f"MLflow database does not exist: {database_path}")

    query = """
        SELECT p.value, lm.value
        FROM experiments AS e
        JOIN runs AS r
          ON r.experiment_id = e.experiment_id
        JOIN params AS p
          ON p.run_uuid = r.run_uuid
         AND p.key = ?
        JOIN latest_metrics AS lm
          ON lm.run_uuid = r.run_uuid
         AND lm.key = ?
        WHERE e.name = ?
          AND e.lifecycle_stage = 'active'
          AND r.lifecycle_stage = 'active'
          AND lm.is_nan = 0
    """

    try:
        with sqlite3.connect(database_path) as connection:
            experiment = connection.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?",
                (experiment_name,),
            ).fetchone()
            if experiment is None:
                raise ValueError(
                    f"MLflow experiment not found: {experiment_name!r}"
                )
            rows = connection.execute(
                query,
                (WINDOW_PARAMETER, F1_METRIC, experiment_name),
            ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"could not read MLflow database: {error}") from error

    values_by_window: dict[float, list[float]] = defaultdict(list)
    for raw_window, raw_f1 in rows:
        try:
            window_seconds = float(raw_window)
            f1 = float(raw_f1)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "encountered a non-numeric window_seconds or "
                "collector_holdout.macro_f1 value"
            ) from error
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError(f"invalid window_seconds value: {raw_window!r}")
        if not math.isfinite(f1) or not 0 <= f1 <= 1:
            raise ValueError(f"invalid {F1_METRIC} value: {raw_f1!r}")
        values_by_window[window_seconds].append(f1)

    if not values_by_window:
        raise ValueError(
            f"experiment {experiment_name!r} has no active runs containing both "
            f"{WINDOW_PARAMETER!r} and {F1_METRIC!r}"
        )

    return [
        WindowF1Average(
            window_seconds=window,
            average_f1=sum(values) / len(values),
            run_count=len(values),
        )
        for window, values in sorted(values_by_window.items())
    ]


def _format_window_seconds(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def build_figure(averages: Sequence[WindowF1Average]):
    """Build the bar chart using a distinct Viridis color for each group."""
    if not averages:
        raise ValueError("at least one window-size average is required")

    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure_width = max(7.5, len(averages) * 1.1 + 2.5)
    figure = Figure(figsize=(figure_width, 5.5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    palette = colormaps["viridis"].resampled(len(averages))
    bars = axis.bar(
        [_format_window_seconds(item.window_seconds) for item in averages],
        [item.average_f1 for item in averages],
        color=[palette(index) for index in range(len(averages))],
    )
    axis.set_title("Average Collector Holdout Macro F1 by Window Duration")
    axis.set_xlabel("Window Seconds")
    axis.set_ylabel("Average Macro F1 Score")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.bar_label(bars, fmt="%.4f", padding=3)
    figure.tight_layout()
    return figure


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot average collector_holdout.macro_f1 by window_seconds for an "
            "MLflow experiment."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"MLflow SQLite database (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT!r})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"figure destination (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        averages = load_window_f1_averages(args.database, args.experiment)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / OUTPUT_FILENAME
        figure = build_figure(averages)
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
