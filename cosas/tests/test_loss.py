import pytest

import torch

from cosas.losses import IoULoss, MCCLosswithLogits


@pytest.mark.parametrize(
    "logits, target, expected_loss",
    [
        pytest.param(
            torch.tensor([1e5, 1e5]), torch.tensor([1, 1]), 0.0, id="IoU:All positive"
        ),
        pytest.param(
            torch.tensor([1e5, -1e5]),
            torch.tensor([1, 0]),
            0.0,
            id="IoU:half positive, half negtaive",
        ),
        pytest.param(
            torch.tensor([-1e5, -1e5]),
            torch.tensor([0, 0]),
            0.0,
            id="IoU:All negative",
        ),
        pytest.param(
            torch.tensor([1e5, -1e5, -1e5]),
            torch.tensor([1, 0, 0]),
            0.0,
            id="IoU:1P+2N",
        ),
    ],
)
def test_iou_loss(logits, target, expected_loss):
    loss = IoULoss()
    assert loss(logits, target) == expected_loss


@pytest.mark.parametrize(
    "logits, target, expected_loss",
    [
        pytest.param(
            torch.tensor([1e5, 1e5]), torch.tensor([1, 1]), 1.0, id="MCC:All positive"
        ),
        pytest.param(
            torch.tensor([1e5, -1e5]), torch.tensor([1, 0]), 0.5, id="MCC: 1 incorrect"
        ),
        pytest.param(
            torch.tensor([-1e5, -1e5]),
            torch.tensor([0, 0]),
            1.0,
            id="IoU:MCC negative",
        ),
    ],
)
def test_mcc_loss(logits, target, expected_loss):
    loss = MCCLosswithLogits()
    result = loss(logits, target)
    assert result == expected_loss
