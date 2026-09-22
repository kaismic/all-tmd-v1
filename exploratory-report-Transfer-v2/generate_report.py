"""Generate the exploratory report for the ALL-TMD Transfer v2 experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable, Sequence


EXPERIMENT_NAME = "ALL-TMD Transfer v2"
F1_METRIC = "collector_holdout.macro_f1"
EXPECTED_WINDOWS = (10, 20, 30, 60)
EXPECTED_SENSORS = (
    "accelerometer,gyroscope",
    "accelerometer,gyroscope,magnetometer",
    "accelerometer,gyroscope,magnetometer,pressure",
)
WINDOW_FIGURE = "average-collector-holdout-macro-f1-by-window-seconds.png"
SENSOR_FIGURE = "average-collector-holdout-macro-f1-by-sensors.png"
LATEX_REPORT = "all-tmd-transfer-v2-tables.tex"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    config_hash: str
    window_seconds: int
    sensors: str
    macro_f1: float


@dataclass(frozen=True)
class Average:
    label: int | str
    macro_f1: float
    run_count: int


@dataclass(frozen=True)
class SnapshotRow:
    mode: str
    sessions: int
    participants: int
    duration: str
    samples: int


def load_runs(database_path: Path) -> list[RunResult]:
    """Load completed, active runs from the named MLflow experiment."""
    query = """
        SELECT r.run_uuid, config.value, window.value, sensors.value, metric.value
        FROM experiments AS experiment
        JOIN runs AS r
          ON r.experiment_id = experiment.experiment_id
        JOIN params AS config
          ON config.run_uuid = r.run_uuid
         AND config.key = 'config_hash'
        JOIN params AS window
          ON window.run_uuid = r.run_uuid
         AND window.key = 'window_seconds'
        JOIN params AS sensors
          ON sensors.run_uuid = r.run_uuid
         AND sensors.key = 'sensors'
        JOIN latest_metrics AS metric
          ON metric.run_uuid = r.run_uuid
         AND metric.key = ?
        WHERE experiment.name = ?
          AND experiment.lifecycle_stage = 'active'
          AND r.lifecycle_stage = 'active'
          AND r.status = 'FINISHED'
          AND metric.is_nan = 0
        ORDER BY r.start_time, r.run_uuid
    """
    if not database_path.is_file():
        raise ValueError(f"MLflow database does not exist: {database_path}")
    try:
        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(query, (F1_METRIC, EXPERIMENT_NAME)).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"could not read MLflow database: {error}") from error

    runs: list[RunResult] = []
    for run_id, config_hash, raw_window, sensors, raw_f1 in rows:
        try:
            window = int(raw_window)
            macro_f1 = float(raw_f1)
        except (TypeError, ValueError) as error:
            raise ValueError(f"run {run_id} has a non-numeric window or F1") from error
        if not math.isfinite(macro_f1) or not 0 <= macro_f1 <= 1:
            raise ValueError(f"run {run_id} has invalid {F1_METRIC}: {raw_f1!r}")
        runs.append(
            RunResult(
                run_id=str(run_id),
                config_hash=str(config_hash),
                window_seconds=window,
                sensors=str(sensors),
                macro_f1=macro_f1,
            )
        )

    combinations = Counter((run.window_seconds, run.sensors) for run in runs)
    expected = {(window, sensors) for window in EXPECTED_WINDOWS for sensors in EXPECTED_SENSORS}
    if set(combinations) != expected or any(count != 1 for count in combinations.values()):
        raise ValueError(
            "experiment does not contain exactly one finished run for each expected "
            "window/sensor combination"
        )
    return runs


def averages_for(runs: Iterable[RunResult], attribute: str, order: Sequence[int | str]) -> list[Average]:
    values: dict[int | str, list[float]] = defaultdict(list)
    for run in runs:
        values[getattr(run, attribute)].append(run.macro_f1)
    return [
        Average(label=label, macro_f1=sum(values[label]) / len(values[label]), run_count=len(values[label]))
        for label in order
    ]


def duration_seconds(session: dict[str, Any]) -> float | None:
    value = session.get("duration_seconds")
    if value is not None and not isinstance(value, bool):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    for start_key, end_key in (
        ("trimmed_start_ms", "trimmed_end_ms"),
        ("started_at_ms", "stopped_at_ms"),
    ):
        try:
            start = int(session[start_key])
            end = int(session[end_key])
        except (KeyError, TypeError, ValueError):
            continue
        if end >= start:
            return (end - start) / 1000
    return None


def format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_snapshot_sessions(
    runs: Sequence[RunResult], work_root: Path, sessions_root: Path
) -> list[dict[str, Any]]:
    """Recover the common frozen collector snapshot from run checkpoints."""
    checkpoint_sets: list[set[str]] = []
    for run in runs:
        checkpoint = work_root / run.config_hash / "events" / "collector" / "checkpoint.json"
        if not checkpoint.is_file():
            raise ValueError(f"collector checkpoint does not exist: {checkpoint}")
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            raise ValueError(f"collector checkpoint is invalid: {checkpoint}")
        checkpoint_sets.append(set(payload))
    if not checkpoint_sets or any(ids != checkpoint_sets[0] for ids in checkpoint_sets[1:]):
        raise ValueError("experiment runs do not share one collector snapshot")

    expected_ids = checkpoint_sets[0]
    sessions_by_id: dict[str, dict[str, Any]] = {}
    for metadata_path in sessions_root.rglob("*.metadata.json"):
        session = json.loads(metadata_path.read_text(encoding="utf-8"))
        session_id = str(session.get("session_id") or "")
        if session_id not in expected_ids:
            continue
        if session_id in sessions_by_id:
            raise ValueError(f"duplicate metadata for collector session {session_id}")
        sessions_by_id[session_id] = session
    missing = expected_ids - sessions_by_id.keys()
    if missing:
        raise ValueError(f"missing metadata for {len(missing)} collector sessions")
    return [sessions_by_id[session_id] for session_id in sorted(expected_ids)]


def build_snapshot_rows(sessions: Sequence[dict[str, Any]]) -> list[SnapshotRow]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session.get("vehicle_type") or "unknown").strip().lower()].append(session)

    rows: list[SnapshotRow] = []
    for mode, mode_sessions in sorted(grouped.items()):
        rows.append(
            SnapshotRow(
                mode=mode.title(),
                sessions=len(mode_sessions),
                participants=len({str(session.get("participant_id") or "unknown") for session in mode_sessions}),
                duration=format_duration(sum(duration_seconds(session) or 0 for session in mode_sessions)),
                samples=sum(int(session.get("sample_count") or 0) for session in mode_sessions),
            )
        )
    rows.append(
        SnapshotRow(
            mode="Total",
            sessions=len(sessions),
            participants=len({str(session.get("participant_id") or "unknown") for session in sessions}),
            duration=format_duration(sum(duration_seconds(session) or 0 for session in sessions)),
            samples=sum(int(session.get("sample_count") or 0) for session in sessions),
        )
    )
    return rows


def plot_bar_chart(
    averages: Sequence[Average], labels: Sequence[str], title: str, xlabel: str, output_path: Path
) -> None:
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.2, 5.5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    palette = colormaps["viridis"].resampled(len(averages))
    bars = axis.bar(
        labels,
        [average.macro_f1 for average in averages],
        color=[palette(index) for index in range(len(averages))],
        width=0.68,
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Average Collector Holdout Macro F1")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.bar_label(bars, fmt="%.4f", padding=4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    figure.clear()


def render_latex(
    window_averages: Sequence[Average],
    sensor_averages: Sequence[Average],
    snapshot_rows: Sequence[SnapshotRow],
) -> str:
    lines = [
        "% Generated by generate_report.py from MLflow experiment ALL-TMD Transfer v2.",
        "% Macro F1 values are arithmetic means across the indicated finished runs.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Average collector holdout macro F1 by window duration.}",
        r"\label{tab:transfer-v2-window-f1}",
        r"\begin{tabular}{rrr}",
        r"\hline",
        "\\textbf{Window (seconds)} & \\textbf{Average Macro F1} & "
        "\\textbf{Runs} \\\\",
        r"\hline",
    ]
    lines.extend(
        f"{average.label} & {average.macro_f1:.4f} & {average.run_count} \\\\" for average in window_averages
    )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Average collector holdout macro F1 by sensor configuration.}",
            r"\label{tab:transfer-v2-sensor-f1}",
            r"\begin{tabular}{lrr}",
            r"\hline",
            "\\textbf{Sensors} & \\textbf{Average Macro F1} & "
            "\\textbf{Runs} \\\\",
            r"\hline",
        ]
    )
    lines.extend(
        f"{average.label.replace(',', ', ')} & {average.macro_f1:.4f} & {average.run_count} \\\\"
        for average in sensor_averages
    )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            "% The 12 experiment runs shared the same collector checkpoint (104 session IDs).",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Collector data snapshot shared by all ALL-TMD Transfer v2 runs.}",
            r"\label{tab:transfer-v2-collector-snapshot}",
            r"\begin{tabular}{lrrrr}",
            r"\hline",
            "\\textbf{Mode} & \\textbf{Sessions} & \\textbf{Participants} & "
            "\\textbf{Duration} & \\textbf{Samples} \\\\",
            r"\hline",
        ]
    )
    for index, row in enumerate(snapshot_rows):
        if row.mode == "Total" and index:
            lines.append(r"\hline")
        lines.append(
            f"{row.mode} & {row.sessions} & {row.participants} & {row.duration} & {row.samples:,} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("/data/all-tmd-work/mlflow.db"))
    parser.add_argument("--work-root", type=Path, default=Path("/data/all-tmd-work"))
    parser.add_argument("--sessions-root", type=Path, default=Path("/data/downloaded_sessions"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runs = load_runs(args.database)
        window_averages = averages_for(runs, "window_seconds", EXPECTED_WINDOWS)
        sensor_averages = averages_for(runs, "sensors", EXPECTED_SENSORS)
        sessions = load_snapshot_sessions(runs, args.work_root, args.sessions_root)
        snapshot_rows = build_snapshot_rows(sessions)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_bar_chart(
            window_averages,
            [str(value) for value in EXPECTED_WINDOWS],
            "Average Collector Holdout Macro F1 by Window Duration",
            "Window Seconds",
            args.output_dir / WINDOW_FIGURE,
        )
        plot_bar_chart(
            sensor_averages,
            [
                "Accelerometer +\nGyroscope",
                "Accelerometer + Gyroscope +\nMagnetometer",
                "Accelerometer + Gyroscope +\nMagnetometer + Pressure",
            ],
            "Average Collector Holdout Macro F1 by Sensors",
            "Sensors",
            args.output_dir / SENSOR_FIGURE,
        )
        (args.output_dir / LATEX_REPORT).write_text(
            render_latex(window_averages, sensor_averages, snapshot_rows),
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for path in (WINDOW_FIGURE, SENSOR_FIGURE, LATEX_REPORT):
        print(args.output_dir / path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
