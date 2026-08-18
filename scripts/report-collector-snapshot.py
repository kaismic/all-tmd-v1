"""Report the collector snapshot captured for a downloaded AWS run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Sequence


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LOG_SESSION_PATTERN = re.compile(
    r"path=/data/downloaded_sessions/(.+(?:\.json|\.json\.gz))$"
)


def integer_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        start = integer_or_none(session.get(start_key))
        end = integer_or_none(session.get(end_key))
        if start is not None and end is not None and end >= start:
            return (end - start) / 1000
    return None


def format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def session_id_digest(sessions: list[dict[str, Any]]) -> str:
    session_ids: list[str] = []
    for session in sessions:
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Collector snapshot contains an invalid session_id")
        session_ids.append(session_id)
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("Collector snapshot contains duplicate session IDs")
    canonical = json.dumps(
        sorted(session_ids),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_table(rows: list[list[str]]) -> str:
    widths = [
        max(len(row[index]) for row in rows)
        for index in range(len(rows[0]))
    ]
    lines: list[str] = []
    for index, row in enumerate(rows):
        if index > 1 and row[0] == "Total":
            lines.append("  ".join("-" * width for width in widths).rstrip())
        lines.append(
            "  ".join(
                value.ljust(widths[column])
                for column, value in enumerate(row)
            ).rstrip()
        )
        if index == 0:
            lines.append("  ".join("-" * width for width in widths).rstrip())
    return "\n".join(lines)


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


def render_latex_table(rows: list[list[str]]) -> str:
    columns = "l" + "r" * (len(rows[0]) - 1)
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\hline",
    ]
    for index, row in enumerate(rows):
        if index > 1 and row[0] == "Total":
            lines.append(r"\hline")
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
        if index == 0:
            lines.append(r"\hline")
    lines.extend((r"\hline", r"\end{tabular}"))
    return "\n".join(lines)


def load_snapshot(path: Path, run_id: str) -> list[dict[str, Any]]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ValueError(f"Unsupported collector snapshot manifest: {path}")
    if snapshot.get("run_id") != run_id:
        raise ValueError(
            f"Collector snapshot RunId does not match {run_id!r}: {path}"
        )
    sessions = snapshot.get("sessions")
    if not isinstance(sessions, list) or not all(
        isinstance(session, dict) for session in sessions
    ):
        raise ValueError(f"Collector snapshot sessions are invalid: {path}")
    if snapshot.get("session_count") != len(sessions):
        raise ValueError(f"Collector snapshot session_count is invalid: {path}")
    digest = session_id_digest(sessions)
    if snapshot.get("session_id_digest") != digest:
        raise ValueError(f"Collector snapshot session_id_digest is invalid: {path}")
    return sessions


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


def legacy_session_paths(log_path: Path) -> list[PurePosixPath]:
    paths: set[PurePosixPath] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_SESSION_PATTERN.search(line)
        if match:
            paths.add(PurePosixPath(match.group(1)))
    if not paths:
        raise ValueError(f"No collector session paths found in legacy log: {log_path}")
    return sorted(paths, key=str)


def load_legacy_snapshot(log_path: Path, sessions_dir: Path) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    missing: list[Path] = []
    for relative_path in legacy_session_paths(log_path):
        payload_path = sessions_dir.joinpath(*relative_path.parts)
        metadata_path = payload_path.with_suffix(
            f"{payload_path.suffix}.metadata.json"
        )
        if not metadata_path.is_file():
            missing.append(metadata_path)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Session sidecar must be a JSON object: {metadata_path}")
        sessions.append(metadata)
    if missing:
        raise ValueError(
            f"{len(missing)} historical session sidecar(s) are unavailable; "
            f"first missing path: {missing[0]}"
        )
    return sessions


def build_mode_rows(sessions: list[dict[str, Any]]) -> list[list[str]]:
    counts: Counter[str] = Counter()
    participants: dict[str, set[str]] = defaultdict(set)
    durations: Counter[str] = Counter()
    samples: Counter[str] = Counter()
    all_participants: set[str] = set()
    for session in sessions:
        mode = str(session.get("vehicle_type") or "unknown").strip().lower()
        participant = str(session.get("participant_id") or "unknown")
        counts[mode] += 1
        participants[mode].add(participant)
        all_participants.add(participant)
        durations[mode] += duration_seconds(session) or 0
        samples[mode] += integer_or_none(session.get("sample_count")) or 0

    rows = [["Mode", "Sessions", "Participants", "Duration", "Samples"]]
    for mode in sorted(counts):
        rows.append(
            [
                mode,
                str(counts[mode]),
                str(len(participants[mode])),
                format_duration(durations[mode]),
                f"{samples[mode]:,}",
            ]
        )
    rows.append(
        [
            "Total",
            str(len(sessions)),
            str(len(all_participants)),
            format_duration(sum(durations.values())),
            f"{sum(samples.values()):,}",
        ]
    )
    return rows


def build_participant_duration_rows(
    sessions: list[dict[str, Any]],
) -> list[list[str]]:
    modes = sorted(
        {str(session.get("vehicle_type") or "unknown").strip().lower() for session in sessions}
    )
    participants = sorted(
        {str(session.get("participant_id") or "unknown") for session in sessions}
    )
    durations: Counter[tuple[str, str]] = Counter()
    for session in sessions:
        participant = str(session.get("participant_id") or "unknown")
        mode = str(session.get("vehicle_type") or "unknown").strip().lower()
        durations[(participant, mode)] += duration_seconds(session) or 0

    rows = [["Participant", *modes, "Total"]]
    for participant in participants:
        values = [durations[(participant, mode)] for mode in modes]
        rows.append(
            [
                participant,
                *[format_duration(value) if value else "" for value in values],
                format_duration(sum(values)),
            ]
        )
    rows.append(
        [
            "Total",
            *[
                format_duration(
                    sum(durations[(participant, mode)] for participant in participants)
                )
                for mode in modes
            ],
            format_duration(sum(durations.values())),
        ]
    )
    return rows


def render_report(
    run_id: str,
    sessions: list[dict[str, Any]],
    snapshot_source: str,
    *,
    latex: bool = False,
) -> str:
    known_durations = sum(duration_seconds(session) is not None for session in sessions)
    summary = [
        ["Metric", "Value"],
        ["Run ID", run_id],
        ["Snapshot source", snapshot_source],
        ["Session payloads", str(len(sessions))],
        ["Session ID digest", session_id_digest(sessions)],
        ["Sessions with duration", str(known_durations)],
    ]
    tables = (
        ("Collector Snapshot Summary", summary),
        ("Transport Mode Summary", build_mode_rows(sessions)),
        (
            "Total Uploaded Session Duration vs Participants",
            build_participant_duration_rows(sessions),
        ),
    )
    if latex:
        return "\n\n".join(
            rf"\subsection*{{{latex_escape(title)}}}\n{render_latex_table(rows)}"
            for title, rows in tables
        )
    return "\n\n".join(
        title + "\n" + render_table(rows) for title, rows in tables
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Display the collector payload snapshot captured for a downloaded "
            "All-TMD AWS RunId."
        )
    )
    parser.add_argument("run_id")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print the report tables as LaTeX tabular environments.",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        help=(
            "Downloaded collector directory used only to reconstruct legacy runs "
            "that predate collector-snapshot.json."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        print("error: RunId contains unsupported characters.", file=sys.stderr)
        return 2

    project_dir = Path(__file__).resolve().parents[1]
    results_root = args.results_root or project_dir / "aws-results"
    run_dir = results_root / args.run_id
    if not run_dir.is_dir():
        print(f"error: downloaded run does not exist: {run_dir}", file=sys.stderr)
        return 1

    snapshot_path = run_dir / "run" / "collector-snapshot.json"
    try:
        if snapshot_path.is_file():
            sessions = load_snapshot(snapshot_path, args.run_id)
            source = str(snapshot_path)
        else:
            sessions_dir = args.sessions_dir or configured_sessions_dir(project_dir)
            if sessions_dir is None:
                raise ValueError(
                    "Legacy run has no collector-snapshot.json; pass --sessions-dir."
                )
            sessions = load_legacy_snapshot(run_dir / "run" / "run.log", sessions_dir)
            source = f"legacy run.log plus sidecars in {sessions_dir}"
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(render_report(args.run_id, sessions, source, latex=args.latex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
