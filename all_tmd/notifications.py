from __future__ import annotations

import argparse
import os
import platform
import re
import sys
from collections.abc import Iterable
from urllib import error, parse, request

STEP_EVENTS = frozenset(
    {"ingest-train-dataset", "ingest-collector", "features", "train"}
)
NOTIFICATION_EVENTS = STEP_EVENTS | {"all-trials"}
EVENT_ALIASES = {
    "all": NOTIFICATION_EVENTS,
    "all-trials": {"all-trials"},
    "features": {"features"},
    "ingest": {"ingest-train-dataset", "ingest-collector"},
    "ingest-collector": {"ingest-collector"},
    "ingest-train-dataset": {"ingest-train-dataset"},
    "none": set(),
    "run": {"all-trials"},
    "steps": STEP_EVENTS,
    "train": {"train"},
}


def configured_notification_events(value: str | None = None) -> frozenset[str]:
    configured = os.getenv("NTFY_EVENTS", "steps") if value is None else value
    names = [
        name.lower()
        for name in re.split(r"[\s,]+", configured.strip())
        if name
    ]
    if not names:
        names = ["steps"]

    unknown = sorted(set(names) - EVENT_ALIASES.keys())
    if unknown:
        available = ", ".join(sorted(EVENT_ALIASES))
        raise ValueError(
            f"Unknown NTFY_EVENTS value(s): {', '.join(unknown)}. "
            f"Available values: {available}"
        )

    events: set[str] = set()
    for name in names:
        events.update(EVENT_ALIASES[name])
    return frozenset(events)


def publish_notification(
    task: str,
    exit_code: int,
    duration: float,
    *,
    event: str | None = None,
) -> bool:
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return False

    selected_event = (event or task).strip().lower()
    if selected_event not in NOTIFICATION_EVENTS:
        raise ValueError(f"Unknown notification event: {selected_event}")
    try:
        enabled_events = configured_notification_events()
    except ValueError as exc:
        print(f"Warning: ntfy notification configuration is invalid: {exc}", file=sys.stderr)
        return False
    if selected_event not in enabled_events:
        return False

    server = os.getenv("NTFY_SERVER", "https://ntfy.sh").strip().rstrip("/")
    token = os.getenv("NTFY_TOKEN", "").strip()
    status = "completed" if exit_code == 0 else "failed"
    headers = {
        "Title": f"All-TMD {task} {status}",
        "Priority": "default" if exit_code == 0 else "high",
        "Tags": "heavy_check_mark" if exit_code == 0 else "x",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    message = (
        f"Task '{task}' {status} on {platform.node() or 'Docker'} "
        f"after {_duration(duration)} (exit code {exit_code})."
    )
    notification = request.Request(
        f"{server}/{parse.quote(topic, safe='')}",
        data=message.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(notification, timeout=10):
            pass
    except (error.URLError, OSError) as exc:
        print(f"Warning: could not send ntfy notification: {exc}", file=sys.stderr)
        return False
    return True


def main(arguments: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Publish an All-TMD ntfy event")
    parser.add_argument("task")
    parser.add_argument("--event", required=True, choices=sorted(NOTIFICATION_EVENTS))
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    args = parser.parse_args(arguments)
    publish_notification(
        args.task,
        args.exit_code,
        args.duration_seconds,
        event=args.event,
    )


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
