import argparse
from io import BytesIO
from typing import Tuple

import random
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from .data_model import COSASData, Scanncers
from .transforms import remove_pad, reverse_tesellation


def get_config() -> argparse.ArgumentParser:
    """
    Note:
        num_workers 필요없음. 이미 메모리에 다 올려서 필요없는듯
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run_name", type=str, default="baseline", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--n_patience", type=int, default=10, help="Number of patience epochs"
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def train_val_split(
    cosas_data: COSASData,
    train_val_test: Tuple[float, float, float],
    random_seed: int = 42,
) -> Tuple:
    train_images = list()
    val_images = list()
    test_images = list()

    train_masks = list()
    val_masks = list()
    test_masks = list()

    train_size, val_size, test_size = train_val_test
    val_size = val_size / (train_size + val_size)
    for scanner in Scanncers:
        scanner_data = getattr(cosas_data, scanner.name)
        images = scanner_data.images
        masks = scanner_data.masks

        # slide using random indices
        train_val_images, test_images, train_val_masks, test_masks = train_test_split(
            images, masks, test_size=test_size, random_state=random_seed
        )
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_val_images,
            train_val_masks,
            test_size=val_size,
            random_state=random_seed,
        )

        train_images.extend(train_images)
        train_masks.extend(train_masks)
        val_images.extend(val_images)
        val_masks.extend(val_masks)
        test_images.extend(test_images)
        test_masks.extend(test_masks)

    return (
        (train_images, train_masks),
        (val_images, val_masks),
        (test_images, test_masks),
    )


def plot_patch_xypred(
    original_x: np.ndarray,
    original_y: np.ndarray,
    pred_masks: np.ndarray,
):
    """patches을 다시 original size으로 concat하여 이미지를 시각화함

    Note:
        어짜피 test time에 사용할거여서 original x, y받아도 될듯

    Args:
        pred_y (torch.Tensor): predicted class, or confidences
                               (N, 224, 224, 1)
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_x)
    axes[0].set_title("Input X")

    axes[1].imshow(original_y, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_masks, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")

    plt.tight_layout()

    return fig, axes


def plot_xypred(
    original_x: np.ndarray,
    original_y: np.ndarray,
    pred_masks: torch.Tensor,
):
    """patches을 다시 original size으로 concat하여 이미지를 시각화함

    Note:
        어짜피 test time에 사용할거여서 original x, y받아도 될듯

    Args:
        pred_y (torch.Tensor): predicted class, or confidences
                               (N, 224, 224, 1)
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    original_size = original_y.shape

    to_original = lambda x: remove_pad(
        reverse_tesellation(x, original_size), original_size
    )

    axes[0].imshow(original_x)
    axes[0].set_title("Input X")

    axes[1].imshow(original_y, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    pred_y = to_original(pred_masks.permute(0, 3, 2, 1))
    axes[2].imshow(ToPILImage()(pred_y), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")

    plt.tight_layout()

    return fig, axes
