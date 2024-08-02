import numpy as np
import pytest
from cosas.metrics import calculate_metrics, specificity_score


def test_calculate_metrics1():
    confidences = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    calculate_metrics(confidences, targets)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        pytest.param(np.array([0, 0, 0, 0]), np.array([0, 0, 1, 0]), 0.75, id="TEST1"),
        pytest.param(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 1, id="TEST2"),
        pytest.param(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), 0, id="TEST3"),
        pytest.param(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), 0, id="TEST4"),
        pytest.param(np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), 0, id="TEST5"),
    ],
)
def test_specificity_score(y_true, y_pred, expected):
    assert specificity_score(y_true, y_pred) == expected
