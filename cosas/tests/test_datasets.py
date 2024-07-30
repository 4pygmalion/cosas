import pytest
import numpy as np
from cosas.datasets import SupConDataset


@pytest.mark.parametrize(
    "mask, expected",
    [
        pytest.param(
            np.zeros((3, 32, 32), dtype=np.float32),
            False,
        ),
        pytest.param(
            np.ones((3, 32, 32), dtype=np.float32),
            True,
        ),
    ],
)
def test_annotate_weakly_label(mask, expected):
    transform = lambda x: x
    dataset = SupConDataset([], [], transform=transform)

    assert expected == dataset.annotate_weakly_label(mask)
