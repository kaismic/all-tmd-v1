from pathlib import Path

import yaml


def test_cloudformation_template_is_well_formed_yaml():
    template = Path(__file__).parents[1] / "aws" / "cloudformation.yaml"

    document = yaml.compose(template.read_text(encoding="utf-8"))

    assert document is not None
