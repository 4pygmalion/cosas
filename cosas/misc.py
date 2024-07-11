import argparse
from typing import Tuple

import random
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from .data_model import COSASData, Scanncers


def get_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--run_name", type=str, default="baseline", help="Run name")
    parser.add_argument(
        "--n_patience", type=int, default=7, help="Number of patience epochs"
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
    cosas_data: COSASData, train_val_test: Tuple[float, float, float]
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
            images, masks, test_size=test_size, random_state=42
        )
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_val_images, train_val_masks, test_size=val_size, random_state=42
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
