from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from urllib import error, parse, request

from all_tmd.progress import progress


def publish_notification(task: str, exit_code: int, duration: float) -> None:
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return
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
        f"Docker Compose task '{task}' {status} on "
        f"{platform.node() or 'Docker'} after {_duration(duration)} "
        f"(exit code {exit_code})."
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise SystemExit("A command to run is required")
    started = time.monotonic()
    exit_code = 1
    try:
        progress(f"Docker task starting: {args.task}")
        exit_code = subprocess.run(args.command, check=False).returncode
    except KeyboardInterrupt:
        exit_code = 130
    except OSError as exc:
        print(f"Could not start task command: {exc}", file=sys.stderr)
        exit_code = 127
    finally:
        publish_notification(args.task, exit_code, time.monotonic() - started)
    raise SystemExit(exit_code)


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
