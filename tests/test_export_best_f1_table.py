import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export-best-f1-table.py"
SPEC = importlib.util.spec_from_file_location("export_best_f1_table", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_metrics(
    results_root: Path,
    *,
    download_id: str,
    run_id: str,
    macro_f1: float,
    mode_f1: dict[str, float],
) -> None:
    metrics_dir = (
        results_root
        / download_id
        / "mlflow"
        / "mlartifacts"
        / "1"
        / run_id
        / "artifacts"
    )
    metrics_dir.mkdir(parents=True)
    metrics = {
        "collector_holdout": {
            "macro_f1": macro_f1,
            "classification_report": {
                **{
                    mode: {"f1-score": f1_score}
                    for mode, f1_score in mode_f1.items()
                },
                "accuracy": 0.8,
                "macro avg": {"f1-score": macro_f1},
                "weighted avg": {"f1-score": 0.8},
            },
        }
    }
    (metrics_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )


def test_main_prints_top_three_unique_trials_across_downloads(tmp_path, capsys):
    _write_metrics(
        tmp_path,
        download_id="download-one",
        run_id="a" * 32,
        macro_f1=0.81,
        mode_f1={"bus": 0.71, "car": 0.82, "train": 0.90},
    )
    _write_metrics(
        tmp_path,
        download_id="download-one",
        run_id="b" * 32,
        macro_f1=0.94,
        mode_f1={"bus": 0.93, "car": 0.94, "train": 0.95},
    )
    _write_metrics(
        tmp_path,
        download_id="download-two",
        run_id="c" * 32,
        macro_f1=0.90,
        mode_f1={"bus": 0.89, "car": 0.90, "train": 0.91},
    )
    _write_metrics(
        tmp_path,
        download_id="download-two",
        run_id="d" * 32,
        macro_f1=0.92,
        mode_f1={"bus": 0.91, "car": 0.92, "train": 0.93},
    )
    _write_metrics(
        tmp_path,
        download_id="download-two",
        run_id="b" * 32,
        macro_f1=0.94,
        mode_f1={"bus": 0.93, "car": 0.94, "train": 0.95},
    )

    exit_code = MODULE.main([], results_root=tmp_path)

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "\\caption{Best F1 Score}\n"
        "\\begin{tabular}{c|ccc|c}\n"
        "    \\hline\n"
        "    Run ID & bus & car & train & average \\\\\n"
        "    \\hline\n"
        f"    {'b' * 32} & 0.9300 & 0.9400 & 0.9500 & 0.9400 \\\\\n"
        f"    {'d' * 32} & 0.9100 & 0.9200 & 0.9300 & 0.9200 \\\\\n"
        f"    {'c' * 32} & 0.8900 & 0.9000 & 0.9100 & 0.9000 \\\\\n"
        "    \\hline\n"
        "\\end{tabular}\n"
    )


def test_collect_best_trials_rejects_conflicting_duplicate_run(tmp_path):
    _write_metrics(
        tmp_path,
        download_id="download-one",
        run_id="a" * 32,
        macro_f1=0.8,
        mode_f1={"bus": 0.8},
    )
    _write_metrics(
        tmp_path,
        download_id="download-two",
        run_id="a" * 32,
        macro_f1=0.9,
        mode_f1={"bus": 0.9},
    )

    try:
        MODULE.collect_best_trials(tmp_path, 3)
    except ValueError as error:
        assert "conflicting metrics" in str(error)
    else:
        raise AssertionError("expected conflicting duplicate metrics to be rejected")


def test_main_reports_when_no_downloaded_metrics_exist(tmp_path, capsys):
    exit_code = MODULE.main([], results_root=tmp_path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "no downloaded MLflow metrics.json files" in captured.err
