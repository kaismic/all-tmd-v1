import importlib.util
from pathlib import Path
import sqlite3
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "generate-window-f1-chart.py"
)
SPEC = importlib.util.spec_from_file_location("generate_window_f1_chart", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE experiments (
                experiment_id INTEGER PRIMARY KEY,
                name TEXT,
                lifecycle_stage TEXT
            );
            CREATE TABLE runs (
                run_uuid TEXT PRIMARY KEY,
                experiment_id INTEGER,
                lifecycle_stage TEXT
            );
            CREATE TABLE params (key TEXT, value TEXT, run_uuid TEXT);
            CREATE TABLE latest_metrics (
                key TEXT,
                value REAL,
                is_nan INTEGER,
                run_uuid TEXT
            );
            INSERT INTO experiments VALUES (1, 'ALL-TMD Transfer v2', 'active');
            INSERT INTO experiments VALUES (2, 'Other Experiment', 'active');
            """
        )


def _insert_run(
    database_path: Path,
    run_id: str,
    window_seconds: str,
    macro_f1: float,
    *,
    experiment_id: int = 1,
    lifecycle_stage: str = "active",
) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            (run_id, experiment_id, lifecycle_stage),
        )
        connection.execute(
            "INSERT INTO params VALUES (?, ?, ?)",
            ("window_seconds", window_seconds, run_id),
        )
        connection.execute(
            "INSERT INTO latest_metrics VALUES (?, ?, ?, ?)",
            ("collector_holdout.macro_f1", macro_f1, 0, run_id),
        )


def test_loads_groups_and_averages_active_experiment_runs(tmp_path):
    database_path = tmp_path / "mlflow.db"
    _create_database(database_path)
    _insert_run(database_path, "run-a", "20", 0.8)
    _insert_run(database_path, "run-b", "10", 0.7)
    _insert_run(database_path, "run-c", "20", 0.9)
    _insert_run(database_path, "deleted", "20", 0.1, lifecycle_stage="deleted")
    _insert_run(database_path, "other", "20", 0.2, experiment_id=2)

    averages = MODULE.load_window_f1_averages(database_path)

    assert [(item.window_seconds, item.run_count) for item in averages] == [
        (10.0, 1),
        (20.0, 2),
    ]
    assert [item.average_f1 for item in averages] == pytest.approx([0.7, 0.85])


def test_figure_uses_viridis_colors_and_four_decimal_labels():
    averages = [
        MODULE.WindowF1Average(10.0, 0.81234, 2),
        MODULE.WindowF1Average(20.0, 0.92345, 2),
    ]

    figure = MODULE.build_figure(averages)
    try:
        axis = figure.axes[0]
        assert axis.get_title() == (
            "Average Collector Holdout Macro F1 by Window Duration"
        )
        assert axis.get_xlabel() == "Window Seconds"
        assert [text.get_text() for text in axis.texts] == ["0.8123", "0.9234"]
        assert len({bar.get_facecolor() for bar in axis.patches}) == 2
    finally:
        figure.clear()


def test_main_writes_png_to_requested_output_directory(tmp_path, capsys):
    database_path = tmp_path / "mlflow.db"
    output_dir = tmp_path / "model-metric-figures"
    _create_database(database_path)
    _insert_run(database_path, "run-a", "30", 0.88)

    exit_code = MODULE.main(
        ["--database", str(database_path), "--output-dir", str(output_dir)]
    )

    output_path = output_dir / MODULE.OUTPUT_FILENAME
    assert exit_code == 0
    assert output_path.is_file()
    assert str(output_path) in capsys.readouterr().out
