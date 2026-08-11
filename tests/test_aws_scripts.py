from pathlib import Path


AWS_SCRIPTS = Path(__file__).parents[1] / "scripts" / "aws"


def test_worker_setup_waits_for_ssm_instead_of_ec2_status_checks():
    for name in ("deploy.ps1", "start-run.ps1"):
        script = (AWS_SCRIPTS / name).read_text(encoding="utf-8")

        assert "Wait-AllTmdSsmOnline" in script
        assert '"instance-status-ok"' not in script


def test_ssm_wait_detects_an_externally_stopped_instance():
    common = (AWS_SCRIPTS / "common.ps1").read_text(encoding="utf-8")

    assert "Get-AllTmdEc2InstanceState" in common
    assert '"stopping", "stopped", "shutting-down", "terminated"' in common
    assert "It may have been stopped externally." in common


def test_downloaded_results_viewer_uses_pinned_local_mlflow_server():
    viewer = (AWS_SCRIPTS / "view-results.ps1").read_text(encoding="utf-8")

    assert "ghcr.io/mlflow/mlflow:v3.14.0" in viewer
    assert '"127.0.0.1:${LocalPort}:5002"' in viewer
    assert "sqlite:////data/all-tmd-work/mlflow.db" in viewer
    assert '"--artifacts-destination", "/data/all-tmd-work/mlartifacts"' in viewer
    assert '"--label", "$viewerLabel=true"' in viewer
    assert '"--rm"' in viewer


def test_downloaded_results_viewer_validates_and_auto_selects_runs():
    viewer = (AWS_SCRIPTS / "view-results.ps1").read_text(encoding="utf-8")

    assert "RunId contains unsupported characters." in viewer
    assert 'Sort-Object LastWriteTime -Descending' in viewer
    assert 'Join-Path $_.FullName "mlflow\\mlflow.db"' in viewer
    assert "MLflow may contain only partial results." in viewer
