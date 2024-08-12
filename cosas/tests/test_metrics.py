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
    "confidences, targets, expected",
    [
        pytest.param(
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 1, 0]),
            {
                "f1": 0.0,
                "acc": 0.75,
                "sen": 0.0,
                "spec": 1,
                "auroc": 0.5,
                "prauc": 0.625,
                "iou": 0.0,
                "dice": 0.0,
            },
            id="TEST1",
        ),
        pytest.param(
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            {
                "f1": 0.0,
                "acc": 1.0,
                "sen": 0.0,
                "spec": 1,
                "auroc": 0.0,
                "prauc": 0.0,
                "iou": 0.0,
                "dice": 0.0,
            },
            id="TEST2",
        ),
        pytest.param(
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 1]),
            {
                "f1": 0.0,
                "acc": 0.0,
                "sen": 0.0,
                "spec": 1,
                "auroc": 0.0,
                "prauc": 1.0,
                "iou": 0.0,
                "dice": 0.0,
            },
            id="TEST3",
        ),
        pytest.param(
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            {
                "f1": 1.0,
                "acc": 1.0,
                "sen": 1.0,
                "spec": 1.0,
                "auroc": 0.0,
                "prauc": 1.0,
                "iou": 1.0,
                "dice": 1.0,
            },
            id="TEST4",
        ),
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([1, 1, 0, 0]),
            {
                "f1": 0.5,
                "acc": 0.5,
                "sen": 0.5,
                "spec": 0.5,
                "auroc": 0.5,
                "prauc": 0.625,
                "iou": 0.33,
                "dice": 0.5,
            },
            id="TEST5",
        ),
    ],
)
def test_calculate_metrics(confidences, targets, expected):
    result = calculate_metrics(confidences, targets)
    for metric, value in expected.items():
        diff = value - result[metric]
        pytest.approx(diff, abs=1e-3)  # 2 decimal places
