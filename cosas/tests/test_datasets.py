import pytest
import torch
from cosas.datasets import SupConDataset


@pytest.mark.parametrize(
    "image, mask, expected",
    [
        pytest.param(
            torch.randn(3, 32, 32, dtype=torch.float32),
            torch.zeros(3, 32, 32, dtype=torch.float32),
            torch.tensor([0], dtype=torch.float32),
        ),
        pytest.param(
            torch.randn(3, 32, 32, dtype=torch.float32),
            torch.ones(3, 32, 32, dtype=torch.float32),
            torch.tensor([1], dtype=torch.float32),
        ),
    ],
)
def test_annotate_weakly_label(image, mask, expected):
    transform = lambda x: x
    dataset = SupConDataset([], [], transform=transform)

    assert expected == dataset.annotate_weakly_label(image, mask)
