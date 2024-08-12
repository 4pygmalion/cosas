import numpy as np
import pytest
from cosas.metrics import calculate_metrics, specificity_score


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        pytest.param(np.array([0, 0, 0, 0]), np.array([0, 0, 1, 0]), 0.75, id="TEST1"),
        pytest.param(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 1, id="TEST2"),
        pytest.param(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), 0, id="TEST3"),
        pytest.param(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), 0, id="TEST4"),
    ],
)
def test_specificity_score(y_true, y_pred, expected):
    assert specificity_score(y_true, y_pred) == expected


@pytest.mark.parametrize(
    "confidences, targets",
    [
        pytest.param(np.array([0, 0, 0, 0]), np.array([0, 0, 1, 0]), id="TEST1"),
        pytest.param(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), id="TEST2"),
        pytest.param(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), id="TEST3"),
        pytest.param(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), id="TEST4"),
    ],
)
def test_calculate_metrics(confidences, targets):
    calculate_metrics(confidences, targets)