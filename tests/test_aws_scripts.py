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
