"""Generate tables and figures for one downloaded AWS experiment run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sqlite3
import sys
from typing import Any, Iterable, Sequence


EXPERIMENT_NAME = "ALL-TMD"
SUMMARY_REPORT_KEYS = {"accuracy", "macro avg", "weighted avg"}
TABLES_FILENAME = "aws-run-tables.tex"
WINDOW_FIGURE = "average-collector-holdout-macro-f1-by-window-seconds.png"
LOG_SESSION_PATTERN = re.compile(
    r"path=/data/downloaded_sessions/(.+(?:\.json|\.json\.gz))$"
)


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


@dataclass(frozen=True)
class CollectorSnapshot:
    sessions: tuple[dict[str, Any], ...]
    source: str
    session_id_digest: str


@dataclass(frozen=True)
class SnapshotRow:
    mode: str
    sessions: int
    participants: int
    duration: str
    samples: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _integer_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _duration_seconds(session: dict[str, Any]) -> float | None:
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
        start = _integer_or_none(session.get(start_key))
        end = _integer_or_none(session.get(end_key))
        if start is not None and end is not None and end >= start:
            return (end - start) / 1000
    return None


def _format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _session_id_digest(sessions: Sequence[dict[str, Any]]) -> str:
    session_ids: list[str] = []
    for session in sessions:
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("collector snapshot contains an invalid session_id")
        session_ids.append(session_id)
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("collector snapshot contains duplicate session IDs")
    canonical = json.dumps(
        sorted(session_ids), ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def configured_sessions_dir(project_dir: Path) -> Path | None:
    configured = os.environ.get("ALL_TMD_DATA_DIR")
    env_path = project_dir / ".env"
    if not configured and env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ALL_TMD_DATA_DIR="):
                configured = line.partition("=")[2].strip()
                break
    if not configured:
        return None
    return Path(configured).expanduser() / "downloaded_sessions"


def _load_manifest_snapshot(snapshot_path: Path, run_id: str) -> CollectorSnapshot:
    payload = _read_json(snapshot_path)
    if payload.get("schema_version") != 1 or payload.get("run_id") != run_id:
        raise ValueError(f"collector snapshot manifest is invalid: {snapshot_path}")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not all(
        isinstance(session, dict) for session in sessions
    ):
        raise ValueError(f"collector snapshot sessions are invalid: {snapshot_path}")
    if payload.get("session_count") != len(sessions):
        raise ValueError(f"collector snapshot session_count is invalid: {snapshot_path}")
    digest = _session_id_digest(sessions)
    if payload.get("session_id_digest") != digest:
        raise ValueError(f"collector snapshot session_id_digest is invalid: {snapshot_path}")
    return CollectorSnapshot(tuple(sessions), "captured collector-snapshot.json", digest)


def _legacy_session_paths(log_path: Path) -> list[PurePosixPath]:
    paths: set[PurePosixPath] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_SESSION_PATTERN.search(line)
        if match:
            paths.add(PurePosixPath(match.group(1)))
    if not paths:
        raise ValueError(f"no collector session paths found in legacy log: {log_path}")
    return sorted(paths, key=str)


def _load_legacy_snapshot(log_path: Path, sessions_dir: Path) -> CollectorSnapshot:
    sessions: list[dict[str, Any]] = []
    missing: list[Path] = []
    for relative_path in _legacy_session_paths(log_path):
        payload_path = sessions_dir.joinpath(*relative_path.parts)
        metadata_path = payload_path.with_suffix(f"{payload_path.suffix}.metadata.json")
        if not metadata_path.is_file():
            missing.append(metadata_path)
            continue
        metadata = _read_json(metadata_path)
        expected_session_id = relative_path.name.removesuffix(".gz").removesuffix(".json")
        if metadata.get("session_id") != expected_session_id:
            raise ValueError(
                f"collector metadata session ID does not match its path: {metadata_path}"
            )
        sessions.append(metadata)
    if missing:
        raise ValueError(
            f"{len(missing)} historical collector metadata sidecars are missing; "
            f"first missing path: {missing[0]}"
        )
    digest = _session_id_digest(sessions)
    return CollectorSnapshot(
        tuple(sessions),
        "legacy run.log plus matching collector metadata sidecars",
        digest,
    )


def load_collector_snapshot(
    run_dir: Path, sessions_dir: Path | None
) -> CollectorSnapshot:
    run_id = run_dir.name
    snapshot_path = run_dir / "run" / "collector-snapshot.json"
    if snapshot_path.is_file():
        return _load_manifest_snapshot(snapshot_path, run_id)
    if sessions_dir is None:
        raise ValueError(
            "legacy run has no collector-snapshot.json and no collector sessions "
            "directory is configured"
        )
    return _load_legacy_snapshot(run_dir / "run" / "run.log", sessions_dir)


def build_snapshot_rows(snapshot: CollectorSnapshot) -> list[SnapshotRow]:
    counts: Counter[str] = Counter()
    participants: dict[str, set[str]] = defaultdict(set)
    durations: Counter[str] = Counter()
    samples: Counter[str] = Counter()
    all_participants: set[str] = set()
    for session in snapshot.sessions:
        mode = str(session.get("vehicle_type") or "unknown").strip().lower()
        participant = str(session.get("participant_id") or "unknown")
        counts[mode] += 1
        participants[mode].add(participant)
        all_participants.add(participant)
        durations[mode] += _duration_seconds(session) or 0
        samples[mode] += _integer_or_none(session.get("sample_count")) or 0

    rows = [
        SnapshotRow(
            mode=mode.title(),
            sessions=counts[mode],
            participants=len(participants[mode]),
            duration=_format_duration(durations[mode]),
            samples=samples[mode],
        )
        for mode in sorted(counts)
    ]
    rows.append(
        SnapshotRow(
            mode="Total",
            sessions=len(snapshot.sessions),
            participants=len(all_participants),
            duration=_format_duration(sum(durations.values())),
            samples=sum(samples.values()),
        )
    )
    return rows


def derive_effective_snapshot(
    raw_snapshot: CollectorSnapshot,
    trials: Sequence[TrialResult],
) -> CollectorSnapshot:
    """Recover common effective membership from MLflow count and digest params."""
    summaries = {
        (
            trial.params.get("collector_session_digest"),
            trial.params.get("collector_session_count"),
        )
        for trial in trials
    }
    if len(summaries) != 1:
        raise ValueError(
            "trials do not share one effective collector session count and digest"
        )
    raw_digest, raw_count = summaries.pop()
    if not raw_digest or not raw_count:
        raise ValueError("MLflow trials do not record effective collector membership")
    try:
        expected_count = int(raw_count)
    except ValueError as error:
        raise ValueError(
            f"MLflow collector_session_count is invalid: {raw_count!r}"
        ) from error

    labels = {label for trial in trials for label in trial.class_f1}
    candidates = [
        session
        for session in raw_snapshot.sessions
        if str(session.get("vehicle_type") or "").strip().lower() in labels
    ]
    excluded_count = len(candidates) - expected_count
    if excluded_count < 0:
        raise ValueError(
            f"raw snapshot contains only {len(candidates)} labeled sessions, "
            f"but MLflow records {expected_count}"
        )
    combination_count = math.comb(len(candidates), excluded_count)
    if combination_count > 1_000_000:
        raise ValueError(
            "effective membership cannot be reconstructed safely: matching the "
            f"recorded digest would require testing {combination_count:,} subsets"
        )

    from itertools import combinations

    matches: list[tuple[dict[str, Any], ...]] = []
    for excluded in combinations(candidates, excluded_count):
        excluded_ids = {str(session["session_id"]) for session in excluded}
        included = tuple(
            session
            for session in candidates
            if str(session["session_id"]) not in excluded_ids
        )
        if _session_id_digest(included) == raw_digest:
            matches.append(included)
            if len(matches) > 1:
                break
    if len(matches) != 1:
        raise ValueError(
            "recorded MLflow count and digest did not identify exactly one "
            f"effective subset (matches={len(matches)})"
        )
    return CollectorSnapshot(
        matches[0],
        "unique raw-snapshot subset matching MLflow collector membership",
        raw_digest,
    )


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


def render_snapshot_latex(
    run_id: str,
    trials: Sequence[TrialResult],
    snapshot: CollectorSnapshot | None,
    error: str | None,
    effective_snapshot: CollectorSnapshot | None = None,
    effective_error: str | None = None,
) -> str:
    label_prefix = "aws-run-" + run_id.split("-")[1]
    if snapshot is None:
        explanation = error or "No collector snapshot source was available."
        return "\n".join(
            (
                r"\paragraph{Collector snapshot unavailable.}",
                "The collector payload snapshot could not be reconstructed reliably: "
                + latex_escape(explanation)
                + ".",
                "",
            )
        )

    rows = build_snapshot_rows(snapshot)
    lines = [
        f"% Collector snapshot source: {snapshot.source}.",
        f"% Collector snapshot session ID digest: {snapshot.session_id_digest}.",
    ]
    lines.extend(
        _table(
            "Raw collector payload snapshot at AWS experiment startup.",
            f"tab:{label_prefix}-collector-snapshot",
            ("Mode", "Sessions", "Participants", "Duration", "Samples"),
            [
                (
                    latex_escape(row.mode),
                    str(row.sessions),
                    str(row.participants),
                    row.duration,
                    f"{row.samples:,}",
                )
                for row in rows
            ],
        )
    )
    lines.extend(
        (
            r"\paragraph{Collector snapshot scope.}",
            "The raw payload membership was reconstructed from "
            + latex_escape(snapshot.source)
            + f" and contains {len(snapshot.sessions)} sessions. "
            "This is the collector state fixed at sweep startup, before label, "
            "duration, sampling, and feature-window eligibility filters.",
            "",
        )
    )
    if effective_snapshot is None:
        explanation = effective_error or "No common effective membership was available."
        lines.extend(
            (
                r"\paragraph{Effective collector snapshot unavailable.}",
                "The effective collector membership could not be reconstructed "
                "reliably: "
                + latex_escape(explanation)
                + ".",
                "",
            )
        )
        return "\n".join(lines)

    effective_rows = build_snapshot_rows(effective_snapshot)
    lines.extend(
        (
            f"% Effective collector snapshot source: {effective_snapshot.source}.",
            "% Effective collector snapshot session ID digest: "
            f"{effective_snapshot.session_id_digest}.",
        )
    )
    lines.extend(
        _table(
            "Effective collector session snapshot used by all MLflow trials.",
            f"tab:{label_prefix}-effective-collector-snapshot",
            ("Mode", "Sessions", "Participants", "Duration", "Samples"),
            [
                (
                    latex_escape(row.mode),
                    str(row.sessions),
                    str(row.participants),
                    row.duration,
                    f"{row.samples:,}",
                )
                for row in effective_rows
            ],
        )
    )
    labels = sorted({label.title() for trial in trials for label in trial.class_f1})
    lines.extend(
        (
            r"\paragraph{Effective collector snapshot scope.}",
            f"The {len(effective_snapshot.sessions)}-session effective membership "
            "is the unique subset of the raw snapshot restricted to the configured "
            "transport modes ("
            + latex_escape(", ".join(labels))
            + ") whose canonical session-ID digest matches the value recorded by "
            "every MLflow trial. It therefore reflects label and feature eligibility "
            "filtering before calibration/holdout splitting.",
            "",
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


def generate_report(
    run_dir: Path,
    output_dir: Path,
    sessions_dir: Path | None = None,
) -> list[Path]:
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
    try:
        snapshot = load_collector_snapshot(run_dir, sessions_dir)
        snapshot_error = None
    except (OSError, ValueError, json.JSONDecodeError) as error:
        snapshot = None
        snapshot_error = str(error)
    if snapshot is None:
        effective_snapshot = None
        effective_error = "raw collector snapshot is unavailable"
    else:
        try:
            effective_snapshot = derive_effective_snapshot(snapshot, trials)
            effective_error = None
        except ValueError as error:
            effective_snapshot = None
            effective_error = str(error)

    output_paths: list[Path] = []
    tables_path = output_dir / TABLES_FILENAME
    if kind == "calibration":
        tables = render_calibration_latex(run_id, summary, trials)
        tables += "\n" + render_snapshot_latex(
            run_id,
            trials,
            snapshot,
            snapshot_error,
            effective_snapshot,
            effective_error,
        )
        tables_path.write_text(
            tables, encoding="utf-8"
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
        tables = render_factorial_latex(run_id, summary, trials, kind)
        tables += "\n" + render_snapshot_latex(
            run_id,
            trials,
            snapshot,
            snapshot_error,
            effective_snapshot,
            effective_error,
        )
        tables_path.write_text(
            tables, encoding="utf-8"
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
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        help=(
            "collector payload directory for legacy runs without "
            "collector-snapshot.json (default: ALL_TMD_DATA_DIR/downloaded_sessions)"
        ),
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
    sessions_dir = args.sessions_dir or configured_sessions_dir(project_dir)
    try:
        paths = generate_report(run_dir, output_dir, sessions_dir=sessions_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
