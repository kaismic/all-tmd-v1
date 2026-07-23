from __future__ import annotations

import argparse
import subprocess
import sys
import time

from all_tmd.notifications import publish_notification
from all_tmd.progress import progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event",
        required=True,
        choices=(
            "ingest-train-dataset",
            "ingest-collector",
            "features",
            "train",
        ),
    )
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
        publish_notification(
            args.task,
            exit_code,
            time.monotonic() - started,
            event=args.event,
        )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
