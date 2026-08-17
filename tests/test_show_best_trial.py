import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "show-best-trial.py"
SPEC = importlib.util.spec_from_file_location("show_best_trial", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_metrics(
    results_root: Path,
    *,
    run_id: str,
    trial_hash: str,
    balanced_accuracy: float,
    accuracy: float,
    macro_f1: float,
    recalls: dict[str, float],
) -> None:
    metrics_dir = (
        results_root
        / run_id
        / "work"
        / trial_hash
        / "reports"
        / "nor-tmd"
    )
    metrics_dir.mkdir(parents=True)
    metrics = {
        "config_hash": trial_hash,
        "train_dataset": "nor-tmd",
        "collector_holdout": {
            "balanced_accuracy": balanced_accuracy,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "minimum_class_recall": min(recalls.values()),
            "classification_report": {
                **{
                    mode: {"recall": recall, "support": 10}
                    for mode, recall in recalls.items()
                },
                "accuracy": accuracy,
                "macro avg": {"recall": balanced_accuracy, "support": 30},
                "weighted avg": {"recall": accuracy, "support": 30},
            },
        },
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_find_best_trial_uses_selected_metric_and_returns_mode_recalls(tmp_path):
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="a" * 64,
        balanced_accuracy=0.91,
        accuracy=0.8,
        macro_f1=0.85,
        recalls={"bus": 0.7, "car": 0.8, "train": 0.9},
    )
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="b" * 64,
        balanced_accuracy=0.89,
        accuracy=0.95,
        macro_f1=0.9,
        recalls={"bus": 0.85, "car": 0.95, "train": 0.87},
    )

    result = MODULE.find_best_trial(
        tmp_path / "selected-run", "collector_holdout.balanced_accuracy"
    )

    assert result.trial_hash == "a" * 64
    assert result.selector_value == 0.91
    assert result.recalls == (("bus", 0.7), ("car", 0.8), ("train", 0.9))


def test_main_scopes_search_to_run_id_and_prints_result(tmp_path, capsys):
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="a" * 64,
        balanced_accuracy=0.8,
        accuracy=0.81,
        macro_f1=0.82,
        recalls={"bus": 0.7, "car": 0.8, "train": 0.9},
    )
    _write_metrics(
        tmp_path,
        run_id="other-run",
        trial_hash="b" * 64,
        balanced_accuracy=0.99,
        accuracy=0.99,
        macro_f1=0.99,
        recalls={"bus": 0.99, "car": 0.99, "train": 0.99},
    )

    exit_code = MODULE.main(
        ["collector_holdout.macro_f1", "selected-run"],
        results_root=tmp_path,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "trial name: nor-tmd-aaaaaaaa" in output
    assert "collector_holdout.macro_f1: 0.82" in output
    assert "bus: 0.7" in output
    assert "car: 0.8" in output
    assert "train: 0.9" in output
    assert "bbbbbbbb" not in output


def test_find_best_trial_ignores_mlflow_artifact_copies(tmp_path):
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="a" * 64,
        balanced_accuracy=0.8,
        accuracy=0.81,
        macro_f1=0.82,
        recalls={"bus": 0.7},
    )
    artifact = (
        tmp_path
        / "selected-run"
        / "mlflow"
        / "mlartifacts"
        / "metrics.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    result = MODULE.find_best_trial(
        tmp_path / "selected-run", "collector_holdout.accuracy"
    )

    assert result.trial_hash == "a" * 64


def test_find_best_trial_accepts_minimum_class_recall(tmp_path):
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="a" * 64,
        balanced_accuracy=0.8,
        accuracy=0.8,
        macro_f1=0.8,
        recalls={"bus": 0.7, "car": 0.9, "train": 0.9},
    )
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="b" * 64,
        balanced_accuracy=0.8,
        accuracy=0.8,
        macro_f1=0.8,
        recalls={"bus": 0.8, "car": 0.8, "train": 0.8},
    )

    result = MODULE.find_best_trial(
        tmp_path / "selected-run",
        "collector_holdout.minimum_class_recall",
    )

    assert result.trial_hash == "b" * 64
