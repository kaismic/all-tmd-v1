from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from all_tmd.mlflow_utils import log_confusion_matrix


@pytest.mark.parametrize(
    ("normalize", "expected", "expected_format"),
    [
        (
            False,
            np.array([[3, 1], [2, 4]]),
            "3",
        ),
        (
            True,
            np.array([[0.75, 0.25], [1 / 3, 2 / 3]]),
            "0.75",
        ),
    ],
)
def test_log_confusion_matrix_logs_figure(
    monkeypatch,
    normalize,
    expected,
    expected_format,
):
    logged = {}

    def capture_figure(figure, artifact_file):
        logged["artifact_file"] = artifact_file
        logged["values"] = np.asarray(figure.axes[0].images[0].get_array())
        logged["annotations"] = {
            text.get_text() for text in figure.axes[0].texts
        }

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(log_figure=capture_figure),
    )

    log_confusion_matrix(
        [[3, 1], [2, 4]],
        ["bus", "car"],
        "evaluation/confusion-matrix.png",
        normalize=normalize,
    )

    assert logged["artifact_file"] == "evaluation/confusion-matrix.png"
    np.testing.assert_allclose(logged["values"], expected)
    assert expected_format in logged["annotations"]


def test_log_confusion_matrix_rejects_wrong_dimensions(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(log_figure=lambda *_args: None),
    )

    with pytest.raises(ValueError, match="dimensions"):
        log_confusion_matrix(
            [[1, 2]],
            ["bus", "car"],
            "evaluation/confusion-matrix.png",
        )
