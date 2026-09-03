from pathlib import Path

import yaml


def test_cloudformation_template_is_well_formed_yaml():
    template = Path(__file__).parents[1] / "aws" / "cloudformation.yaml"

    document = yaml.compose(template.read_text(encoding="utf-8"))

    assert document is not None


def test_cloudformation_exposes_v2_shared_worker_contract_and_permissions():
    template = Path(__file__).parents[1] / "aws" / "cloudformation.yaml"
    content = template.read_text(encoding="utf-8")

    assert "ConsumerProjectName:" in content
    assert "Default: all-tmd-v2" in content
    assert "SharedWorkerContractVersion:" in content
    assert "SharedWorkerConsumerProject:" in content
    assert "SharedInputsPrefix:" in content
    assert "${ConsumerProjectName}/config/*" in content
    assert "${ConsumerProjectName}/results/*" in content


def test_cloudformation_can_pin_the_existing_worker_ami_on_updates():
    template = Path(__file__).parents[1] / "aws" / "cloudformation.yaml"
    content = template.read_text(encoding="utf-8")

    assert "WorkerImageId:" in content
    assert "HasExplicitWorkerImageId:" in content
    assert (
        "ImageId: !If [HasExplicitWorkerImageId, !Ref WorkerImageId, "
        "!Ref UbuntuImageId]" in content
    )
