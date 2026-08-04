from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from all_tmd.trial_generator import generate_trials


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REQUIRED_CONFIG_FILES = (
    "model.config.yaml",
    "trial-parameters.json",
)


def create_run_bundle(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    git_repository: str,
    git_commit: str,
    mode: str = "full",
    ntfy_server: str = "https://ntfy.sh",
    ntfy_topic: str = "",
    ntfy_events: str = "all-trials",
    ntfy_token_parameter: str = "/all-tmd-v1/ntfy-token",
    auto_stop: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create the non-secret configuration bundle consumed by the EC2 worker."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "alphanumeric characters, dots, underscores, or hyphens"
        )
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")
    if not git_repository.strip():
        raise ValueError("git_repository must not be empty")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_commit):
        raise ValueError("git_commit must be a full 40-character Git SHA")

    root = Path(project_root)
    destination = Path(output_dir)
    for name in REQUIRED_CONFIG_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required run configuration is missing: {path}")

    parameters_path = root / "trial-parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    generated_trials = generate_trials(parameters)
    trials = generated_trials
    if mode == "smoke":
        smoke_trial = deepcopy(generated_trials[0])
        smoke_trial["training"]["optuna_trials"] = 1
        trials = [smoke_trial]

    destination.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_CONFIG_FILES:
        shutil.copy2(root / name, destination / name)
    trials_path = destination / "trials.json"
    trials_path.write_text(
        json.dumps(trials, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    timestamp = created_at or datetime.now(timezone.utc)
    config_files = (*REQUIRED_CONFIG_FILES, "trials.json")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "created_at": timestamp.astimezone(timezone.utc).isoformat(),
        "git_repository": git_repository,
        "git_commit": git_commit.lower(),
        "trial_count": len(trials),
        "generated_trial_count": len(generated_trials),
        "config_sha256": {
            name: _sha256(destination / name) for name in config_files
        },
        "notifications": {
            "server": ntfy_server,
            "topic": ntfy_topic,
            "events": ntfy_events,
            "token_parameter": ntfy_token_parameter,
        },
        "auto_stop": auto_stop,
    }
    (destination / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an AWS All-TMD run configuration bundle"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-repository", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--ntfy-server", default="https://ntfy.sh")
    parser.add_argument("--ntfy-topic", default="")
    parser.add_argument("--ntfy-events", default="all-trials")
    parser.add_argument(
        "--ntfy-token-parameter",
        default="/all-tmd-v1/ntfy-token",
    )
    parser.add_argument(
        "--no-auto-stop",
        action="store_true",
        help="leave the EC2 instance running after result upload",
    )
    args = parser.parse_args(argv)
    manifest = create_run_bundle(
        args.project_root,
        args.output,
        run_id=args.run_id,
        git_repository=args.git_repository,
        git_commit=args.git_commit,
        mode=args.mode,
        ntfy_server=args.ntfy_server,
        ntfy_topic=args.ntfy_topic,
        ntfy_events=args.ntfy_events,
        ntfy_token_parameter=args.ntfy_token_parameter,
        auto_stop=not args.no_auto_stop,
    )
    print(
        f"Created {manifest['mode']} run bundle {manifest['run_id']} "
        f"with {manifest['trial_count']} trial(s)"
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
