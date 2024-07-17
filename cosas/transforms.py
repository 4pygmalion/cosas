import math
from typing import Tuple, List

import numpy as np
import torch
import albumentations as A
from torchvision.transforms.functional import pad


def get_image_stats(
    images: np.ndarray | List[np.ndarray],
) -> Tuple[tuple, tuple]:
    """
    이미지 배열의 평균 및 표준 편차를 계산합니다.

    이 함수는 입력된 이미지 배열이 4차원 배열인지 확인하고,
    각 차원에 대해 평균 및 표준 편차를 계산하여 반환합니다.

    Params:
        images (np.ndarray): 4차원 이미지 배열 (B, W, H, C)

    Returns:
        Tuple[tuple, tuple]: 이미지 배열의 평균과 표준 편차.

    Exception:
        ValueError: 입력된 이미지 배열이 4차원 배열이 아닐 경우 발생.
    """
    if isinstance(images, list):
        means = list()
        stds = list()
        for image in images:
            means.append(image.mean(axis=(0, 1)))
            stds.append(image.std(axis=(0, 1)))

        means = np.array(means).mean(axis=0)
        stds = np.array(stds).mean(axis=0)

    if isinstance(images, np.ndarray):
        if images.ndim != 4:
            raise ValueError(
                f"Input images array must be 4-dimensional, passed images.ndim({images.ndim})"
            )

        means = np.mean(images, axis=(0, 1, 2))
        stds = np.std(images, axis=(0, 1, 2))

    return tuple(means), tuple(stds)


def pad_image_tensor(
    image_tensor: torch.Tensor, size: tuple = (224, 224)
) -> torch.Tensor:
    """이미지 전체를 주어진 사이즈(size)의 정수배로 만듬

    Params:
        image_tensor (torch.Tensor): shape (C, W, H)
        size

    Returns:
        padded_tensor
    """

    c, image_w, image_h = image_tensor.size()

    patch_w, patch_h = size
    pad_w = (patch_w - (image_w % patch_w)) % patch_w
    pad_h = (patch_h - (image_h % patch_h)) % patch_h

    return pad(image_tensor, (0, 0, pad_h, pad_w))


def remove_pad(image_tensor: torch.Tensor, original_size: tuple):
    """패딩된 이미지 텐서에서 원본 크기로 자르는 함수

    이 함수는 패딩을 추가한 이미지 텐서에서 원본 이미지의 크기만큼 잘라내어
    원본 크기의 이미지 텐서를 반환합니다.

    Args:
        image_tensor (torch.Tensor): 패딩이 추가된 이미지 텐서
        original_size (tuple): 원본 이미지의 크기 (너비, 높이)

    Returns:
        torch.Tensor: 원본 크기로 자른 이미지 텐서
    """
    W, H = original_size
    return image_tensor[:, :W, :H]


def tesellation(image_tensor: torch.Tensor, size: tuple = (224, 224)) -> torch.Tensor:
    """size의 정수배인 image_tensor을 입력받아, 패치단위로 타일링을 진행

    Note:
        아래와 동일한 코드
        >>> patches = []
        >>> for i in range(0, W, patch_w):
        >>>     for j in range(0, H, patch_h):
        >>>         patch = padded_tensor[:, i : i + patch_w, j : j + patch_h]
        >>>         patches.append(patch)
        >>> # 패치들을 텐서로 변환
        >>> patches = torch.stack(patches)

    Returns
        pathces (torch.Tensor): shape (n_patches, c, patch_h, patch_w)

    """

    C, W, H = image_tensor.shape
    patch_w, patch_h = size

    # (C, W, H)
    padded_tensor: torch.Tensor = pad_image_tensor(image_tensor)

    # (C, row, W, h) -> (C, row, w, col, h, w)
    patches = padded_tensor.unfold(1, patch_h, patch_h).unfold(2, patch_w, patch_w)

    return patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, patch_h, patch_w)


def reverse_tesellation(patches: torch.Tensor, original_size: tuple) -> torch.Tensor:
    """
    Tesellation 함수에 의해 생성된 패치들을 병합하여 원래 이미지의 크기로 복원하는 함수입니다.

    Args:
        patches (torch.Tensor): shape (n_patches, C, patch_h, patch_w), 타일링된 패치 텐서
        original_size (tuple): (original_height, original_width), 원본 이미지의 높이와 너비

    Returns:
        torch.Tensor: 병합된 이미지 텐서, shape (C, original_height, original_width)
    """

    # 입력된 패치들의 수, 채널 수, 패치의 높이와 너비를 추출
    _, C, patch_h, patch_w = patches.shape
    original_height, original_width = original_size

    # 패치들을 원래 이미지 크기에 맞게 재배열합니다.
    # 각 차원에 필요한 패치의 수를 계산
    n_patches_h = math.ceil(original_height / patch_h)
    n_patches_w = math.ceil(original_width / patch_w)

    patches = patches.view(n_patches_h, n_patches_w, C, patch_h, patch_w)

    original_image = torch.zeros(C, n_patches_h * patch_h, n_patches_w * patch_w)
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            x_start = i * patch_h
            x_end = x_start + patch_h
            y_start = j * patch_w
            y_end = y_start + patch_w
            original_image[:, x_start:x_end, y_start:y_end] = patches[i, j]

    return original_image


class CopyTransform(A.DualTransform):
    def apply(self, img, **params):
        if any(stride < 0 for stride in img.strides):
            img = np.ascontiguousarray(img)
        return img

    def apply_to_mask(self, mask, **params):
        if any(stride < 0 for stride in mask.strides):
            mask = np.ascontiguousarray(mask)
        return mask

    def get_transform_init_args_names(self):
        return ()
