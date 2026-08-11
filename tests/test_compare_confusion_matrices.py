import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "compare-confusion-matrices.py"
)
SPEC = importlib.util.spec_from_file_location("compare_confusion_matrices", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_metrics(
    results_root: Path,
    *,
    run_name: str,
    trial_hash: str,
    matrix: list[list[int]],
) -> None:
    metrics_dir = (
        results_root
        / run_name
        / "work"
        / trial_hash
        / "reports"
        / "nor-tmd"
    )
    metrics_dir.mkdir(parents=True)
    support = [sum(row) for row in matrix]
    metrics = {
        "config_hash": trial_hash,
        "train_dataset": "nor-tmd",
        "collector_holdout": {
            "classification_report": {
                "bus": {"support": support[0]},
                "car": {"support": support[1]},
                "accuracy": 0.5,
                "macro avg": {"support": sum(support)},
                "weighted avg": {"support": sum(support)},
            },
            "confusion_matrix": matrix,
        },
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_collect_trial_results_normalizes_and_ranks_selected_true_label(tmp_path):
    _write_metrics(
        tmp_path,
        run_name="run-a",
        trial_hash="a" * 64,
        matrix=[[8, 2], [4, 6]],
    )
    _write_metrics(
        tmp_path,
        run_name="run-b",
        trial_hash="b" * 64,
        matrix=[[9, 1], [1, 9]],
    )

    results, available_modes, metrics_count = MODULE.collect_trial_results(
        tmp_path, "bus"
    )

    assert metrics_count == 2
    assert available_modes == {"bus", "car"}
    assert [result.trial_hash for result in results] == ["b" * 64, "a" * 64]
    assert results[0].trial_name == "nor-tmd-bbbbbbbb"
    assert results[0].normalized_true_label_accuracy == 0.9
    assert results[0].normalized_predicted_values == (0.9, 0.1)
    assert results[0].support == 10


def test_main_limits_displayed_trials_and_reports_array_labels(tmp_path, capsys):
    _write_metrics(
        tmp_path,
        run_name="run-a",
        trial_hash="a" * 64,
        matrix=[[8, 2], [4, 6]],
    )
    _write_metrics(
        tmp_path,
        run_name="run-b",
        trial_hash="b" * 64,
        matrix=[[9, 1], [1, 9]],
    )

    exit_code = MODULE.main(["bus", "1"], results_root=tmp_path)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Top 1 of 2 trials" in output
    assert "trial name: nor-tmd-bbbbbbbb" in output
    assert "normalized predicted values [bus, car]: [0.9, 0.1]" in output
    assert "support: 10" in output
    assert "nor-tmd-aaaaaaaa" not in output


def test_collect_trial_results_ignores_mlflow_artifact_copies(tmp_path):
    _write_metrics(
        tmp_path,
        run_name="run-a",
        trial_hash="a" * 64,
        matrix=[[8, 2], [4, 6]],
    )
    artifact = tmp_path / "run-a" / "mlflow" / "mlartifacts" / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    results, _, metrics_count = MODULE.collect_trial_results(tmp_path, "bus")

    assert metrics_count == 1
    assert len(results) == 1
