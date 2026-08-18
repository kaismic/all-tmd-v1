import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export-recall-table.py"
SPEC = importlib.util.spec_from_file_location("export_recall_table", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_metrics(
    results_root: Path,
    *,
    run_id: str,
    trial_hash: str,
    trial_index: int,
    recalls: dict[str, float],
    configured_run_name: str | None = None,
) -> None:
    metrics_dir = (
        results_root / run_id / "work" / trial_hash / "reports" / "nor-tmd"
    )
    metrics_dir.mkdir(parents=True)
    metrics = {
        "config_hash": trial_hash,
        "trial_index": trial_index,
        "train_dataset": "nor-tmd",
        "collector_holdout": {
            "classification_report": {
                **{mode: {"recall": recall} for mode, recall in recalls.items()},
                "accuracy": 0.8,
                "macro avg": {"recall": 0.8},
                "weighted avg": {"recall": 0.8},
            }
        },
    }
    if configured_run_name is not None:
        metrics["run_name"] = configured_run_name
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_main_prints_latex_table_for_every_trial_and_mode(tmp_path, capsys):
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="b" * 64,
        trial_index=2,
        recalls={"car": 0.82345, "bus": 0.71234},
    )
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="a" * 64,
        trial_index=1,
        configured_run_name="baseline_v1",
        recalls={"train": 0.93456, "bus": 0.75678, "car": 0.84567},
    )
    _write_metrics(
        tmp_path,
        run_id="other-run",
        trial_hash="c" * 64,
        trial_index=0,
        recalls={"bus": 0.99},
    )

    exit_code = MODULE.main(
        ["selected-run", "--precision", "3"], results_root=tmp_path
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "\\begin{tabular}{lrrr}\n"
        "\\hline\n"
        "Run & bus & car & train \\\\\n"
        "\\hline\n"
        "baseline\\_v1-nor-tmd-aaaaaaaa & 0.757 & 0.846 & 0.935 \\\\\n"
        "nor-tmd-bbbbbbbb & 0.712 & 0.823 & -- \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )


def test_collect_trial_recalls_ignores_mlflow_artifact_copies(tmp_path):
    _write_metrics(
        tmp_path,
        run_id="selected-run",
        trial_hash="a" * 64,
        trial_index=0,
        recalls={"bus": 0.7},
    )
    artifact = tmp_path / "selected-run" / "mlflow" / "mlartifacts" / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    trials, modes = MODULE.collect_trial_recalls(tmp_path / "selected-run")

    assert len(trials) == 1
    assert modes == ["bus"]


def test_main_rejects_missing_run(tmp_path, capsys):
    exit_code = MODULE.main(["missing-run"], results_root=tmp_path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "downloaded run does not exist" in captured.err
