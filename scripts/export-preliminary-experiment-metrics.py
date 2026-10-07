"""Export eight-column holdout metrics for the four preliminary experiments.

Uses only Python's standard library. Reads original MLflow databases and
per-run artifacts, never generated CSVs or shared work-directory reports.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "experiment_name", "run_id", "transport_mode", "support",
    "precision", "recall", "f1", "accuracy",
)
EXPERIMENTS = (
    ("initial-sensor-window-size", "Initial Sensor and Window Size Experiment", None, 12),
    (
        "pressure-feature-four-transport-classes",
        "Pressure-feature and Four Transport Classes Experiment",
        "20260809T055942Z-9c676a2c-full", 12,
    ),
    (
        "magnetometer-feature", "Magnetometer Feature Experiment",
        "20260812T125950Z-a46cc6a0-full", 12,
    ),
    (
        "calibration-fraction", "Calibration Fraction Experiment",
        "20260814T114054Z-c11f9a1a-full", 3,
    ),
)
SUMMARY_KEYS = {"accuracy", "macro avg", "weighted avg", "micro avg"}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def default_local_work_root() -> Path:
    data_dir = os.environ.get("ALL_TMD_DATA_DIR")
    env_file = ROOT / ".env"
    if not data_dir and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key.strip() == "ALL_TMD_DATA_DIR":
                data_dir = value.strip().strip("\"'")
                break
    data_root = Path(data_dir).expanduser() if data_dir else ROOT / "data"
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    return data_root / "all-tmd-work"


def score(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description}: expected a numeric score")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{description}: score must be finite and between 0 and 1")
    return float(value)


def collect_rows(
    title: str, mlflow_root: Path, expected_runs: int,
    summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    database = mlflow_root / "mlflow.db"
    experiment = "ALL-TMD" if summary is not None else "ALL-TMD Transfer v2"
    # Read-only mode also prevents creating a blank database at a bad path.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        runs = connection.execute(
            """
            SELECT r.run_uuid, r.experiment_id, r.start_time
            FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id
            WHERE e.name = ? AND e.lifecycle_stage = 'active'
              AND r.status = 'FINISHED' AND r.lifecycle_stage = 'active'
              AND EXISTS (
                  SELECT 1 FROM latest_metrics m WHERE m.run_uuid = r.run_uuid
                    AND m.key = 'collector_holdout.macro_f1' AND m.is_nan = 0
              )
            ORDER BY r.start_time, r.run_uuid
            """,
            (experiment,),
        ).fetchall()
    if summary is not None:
        start = datetime.fromisoformat(summary["started_at"]).timestamp() * 1000
        end = datetime.fromisoformat(summary["completed_at"]).timestamp() * 1000
        if start > end:
            raise ValueError(f"{mlflow_root}: invalid AWS execution interval")
        runs = [run for run in runs if start <= run[2] <= end]
    if len(runs) != expected_runs:
        raise ValueError(
            f"{title}: expected {expected_runs} completed runs, found {len(runs)} in {database}"
        )

    trials = []
    expected_modes = {"bus", "car", "train"}
    if summary is not None and summary["run_id"] == EXPERIMENTS[1][2]:
        expected_modes.add("tram")
    for run_id, experiment_id, _ in runs:
        path = mlflow_root / "mlartifacts" / str(experiment_id) / run_id / "artifacts/metrics.json"
        metrics = read_json(path)
        holdout = metrics["collector_holdout"]
        report = holdout["classification_report"]
        modes = set(report) - SUMMARY_KEYS
        if modes != expected_modes:
            raise ValueError(f"{path}: unexpected transport modes: {sorted(modes)}")
        accuracy = score(holdout["accuracy"], f"{path}: accuracy")
        trial_index = metrics["trial_index"]
        if isinstance(trial_index, bool) or not isinstance(trial_index, int):
            raise ValueError(f"{path}: invalid trial_index")
        rows = []
        for mode in sorted(modes):
            values = report[mode]
            support = values["support"]
            if (
                isinstance(support, bool) or not isinstance(support, (int, float))
                or not math.isfinite(support) or support < 0 or int(support) != support
            ):
                raise ValueError(f"{path}: invalid support for {mode}")
            rows.append({
                "experiment_name": title,
                "run_id": run_id,
                "transport_mode": mode,
                "support": int(support),
                "precision": score(values["precision"], f"{path}: {mode} precision"),
                "recall": score(values["recall"], f"{path}: {mode} recall"),
                "f1": score(values["f1-score"], f"{path}: {mode} f1"),
                "accuracy": accuracy,
            })
        if sum(row["support"] for row in rows) != holdout["rows"]:
            raise ValueError(f"{path}: class supports do not sum to holdout rows")
        trials.append((trial_index, run_id, rows))
    return [row for _, _, rows in sorted(trials) for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-work-root", type=Path,
        help=("local all-tmd-work directory (default: ALL_TMD_DATA_DIR from environment "
              "or repository .env, then data/all-tmd-work)"),
    )
    parser.add_argument("--results-root", type=Path, default=ROOT / "aws-results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)
    try:
        local_root = args.local_work_root or default_local_work_root()
        exports = []
        for slug, title, bundle, count in EXPERIMENTS:
            summary = None
            mlflow_root = local_root
            if bundle:
                bundle_root = args.results_root / bundle
                summary = read_json(bundle_root / "run/run-summary.json")
                if summary["run_id"] != bundle or summary["trial_count"] != count:
                    raise ValueError(f"{bundle_root}: summary does not match the expected experiment")
                mlflow_root = bundle_root / "mlflow"
            exports.append((slug, collect_rows(title, mlflow_root, count, summary)))

        # Validate all sources before replacing any existing output file.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for slug, rows in exports:
            path = args.output_dir / f"{slug}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"{path}: {len(rows)} rows")
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
