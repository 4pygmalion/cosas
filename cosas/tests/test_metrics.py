import numpy as np
from cosas.metrics import calculate_metrics


def test_calculate_metrics1():
    confidences = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    calculate_metrics(confidences, targets)


def test_calculate_metrics1():
    confidences = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, np.nan])
    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    calculate_metrics(confidences, targets)
