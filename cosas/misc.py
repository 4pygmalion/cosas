import argparse
from typing import Tuple

import math
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from .data_model import COSASData, Scanncers
from .transforms import remove_pad, reverse_tesellation
from .networks import MODEL_REGISTRY
from .losses import LOSS_REGISTRY


class SetSMPArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "smp", True)
        for info in values:
            key, value = info.split(":")
            setattr(namespace, key.strip(), value.strip())


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
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["patch", "image_mask", "whole", "pre_aug"],
        required=True,
    )
    parser.add_argument(
        "--loss", type=str, choices=list(LOSS_REGISTRY.keys()), required=True
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--n_patience", type=int, default=10, help="Number of patience epochs"
    )
    parser.add_argument("--input_size", type=int, help="Image size", required=True)
    parser.add_argument(
        "--model_name", type=str, help="Model name", choices=list(MODEL_REGISTRY.keys())
    )
    parser.add_argument(
        "--smp",
        nargs="+",
        action=SetSMPArgs,
        help=(
            "Set SMP encoder name and weights \n"
            "For example:"
            "--smp 'encoder_name:efficientnet-b4' 'encoder_weights:imagenet'"
        ),
    )
    parser.add_argument("--use_sn", action="store_true", help="Use stain normalization")
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
        (
            scan_train_val_images,
            scan_test_images,
            scan_train_val_masks,
            scan_test_masks,
        ) = train_test_split(
            images, masks, test_size=test_size, random_state=random_seed
        )
        scan_train_images, scan_val_images, scan_train_masks, scan_val_masks = (
            train_test_split(
                scan_train_val_images,
                scan_train_val_masks,
                test_size=val_size,
                random_state=random_seed,
            )
        )

        train_images.extend(scan_train_images)
        train_masks.extend(scan_train_masks)
        val_images.extend(scan_val_images)
        val_masks.extend(scan_val_masks)
        test_images.extend(scan_test_images)
        test_masks.extend(scan_test_masks)

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





class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
