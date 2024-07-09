from typing import Tuple

import random
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from .data_model import COSASData, Scanncers


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
