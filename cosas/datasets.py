import random
from typing import Tuple, List

import cv2
import tqdm
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor

from cosas.transforms import tesellation, pad_image_tensor


class Patchdataset(Dataset):
    """한 이미지를 패치로 나눠 N개의 패치를 차원을 높여 (1, p, w, h)만든 데이터셋"""

    def __init__(
        self, images, masks, transform: A.Compose | None = None, device: str = "cuda"
    ):
        self.images = images  # uint8
        self.masks = masks  # uint8
        self.transform = transform
        self.device = device

        # uint8
        self.patch_size = (512, 512)
        self.patch_images: List[np.ndarray] = None
        self.patch_masks: List[np.ndarray] = None
        self._tesellate_patch_mask()

    def _tesellate_patch_mask(self):

        self.patch_images = list()
        self.patch_masks = list()
        for image, mask in zip(self.images, self.masks):
            resize_image = cv2.resize(
                image, dsize=(512 * 3, 512 * 3), interpolation=cv2.INTER_NEAREST
            )
            resize_mask = cv2.resize(
                mask[:, :, np.newaxis],
                dsize=(512 * 3, 512 * 3),
                interpolation=cv2.INTER_NEAREST,
            )[:, :, np.newaxis]

            self.patch_images.append(
                tesellation(resize_image, patch_size=self.patch_size).reshape(
                    -1, *self.patch_size, 3
                )
            )
            self.patch_masks.append(
                tesellation(resize_mask, patch_size=self.patch_size).reshape(
                    -1, *self.patch_size, 1
                )
            )

        self.patch_images = np.concatenate(self.patch_images, axis=0)
        self.patch_masks = np.concatenate(self.patch_masks, axis=0)

        return

    def __len__(self):
        return len(self.patch_images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.patch_images[idx]
        mask = self.patch_masks[idx]
        if not self.transform:
            return image, mask

        # self.transform: np.ndarray -> Dict[str, Tensor]
        augmented = self.transform(image=image, mask=mask)  # mask은 normalized 안됨
        image = augmented["image"].to(self.device)
        mask = augmented["mask"].to(self.device)

        return image, mask


class WholeSizeDataset(Dataset):
    def __init__(
        self, images, masks, transform: A.Compose | None = None, device: str = "cuda"
    ):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: float32, float32 type
        """
        image = self.images[idx]
        mask = self.masks[idx]

        # self.transform: np.ndarray -> Dict[str, Tensor]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]  # (Tensor, Tensor)

        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

        patch_images = tesellation(image)  # (B, C, W, H)

        mask = mask.unsqueeze(dim=0)  # (W, H) -> (1, W, H)
        patch_masks = tesellation(mask)
        patch_masks = patch_masks.permute(0, 2, 3, 1)  # (B, C, W, H) -> # (B, W, H, C)

        return patch_images.to(self.device), patch_masks.to(self.device).float()


class ImageMaskDataset(Dataset):
    def __init__(
        self, images, masks, transform: A.Compose | None = None, device="cuda"
    ):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]

        # self.transform: np.ndarray -> Dict[str, Tensor]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

            # stride가 negative인 경우 처리
            if isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask.numpy().copy())
            return image, mask

        else:
            image = torch.from_numpy(image.copy())
            mask = torch.from_numpy(mask.copy())
            return image, mask


class MultiScaleDataset(Dataset):
    """멀티스케일 데이터셋"""

    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        transform: A.Compose | None = None,
        multiples: int = 10,
        image_size: Tuple[int, int] = (512, 512),
        device: str = "cuda",
    ):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.multiples = multiples
        self.image_size = image_size
        self.device = device

        self._pre_augmentation(multiples, image_size=image_size)

    def _pre_augmentation(self, multiples: int = 10, image_size=(512, 512)):
        """이미지, 마스크를 초기화시에 N배를 생성"""

        augmentor = A.RandomResizedCrop(image_size)
        aug_images = list()
        aug_masks = list()
        for iter in range(multiples):
            for image, mask in zip(self.images, self.masks):
                aug = augmentor(image=image, mask=mask)
                aug_images.append(aug["image"])
                aug_masks.append(aug["mask"])

        self.images = self.images + aug_images
        self.masks = self.masks + aug_masks

        return

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"]

        else:
            image = torch.from_numpy(image.copy())
            mask = torch.from_numpy(mask.copy())
            return image, mask


class SupConDataset(Dataset):
    """DataSet for Supervised contrastive learing"""

    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        transform: A.Compose,
        threshold=0.01,
        device="cuda",
        image_size=(386, 386),
    ):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.threshold = threshold
        self.device = device
        self.image_size = image_size
        self._preaug()

    def _preaug(self, n=32):
        self.crop = A.RandomCrop(*self.image_size, p=1)

        images = list()
        masks = list()
        for iter in tqdm.tqdm(range(n), desc="Pre-augmentation"):
            for image, mask in zip(self.images, self.masks):
                aug = self.crop(image=image, mask=mask)
                images.append(aug["image"])
                masks.append(aug["mask"])

        self.images = images
        self.masks = masks

        return

    def __len__(self):
        return len(self.images)

    def annotate_weakly_label(self, mask: np.ndarray) -> bool:
        """Segmentation label -> Image level label"""
        n_pixels: int = np.prod(mask.shape)
        n_positive: int = np.sum(mask)
        positive_ratio = n_positive / n_pixels
        return positive_ratio >= self.threshold

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """한 뷰에서는 동일한 라벨이어야함."""

        image = self.images[idx]
        mask = self.masks[idx]
        aug1 = self.transform(image=image, mask=mask)
        aug2 = self.transform(image=image, mask=mask)
        views = torch.stack([aug1["image"], aug2["image"]], dim=0)

        label = self.annotate_weakly_label(mask)

        return views, torch.tensor([label], dtype=torch.float32)


DATASET_REGISTRY = {
    "patch": Patchdataset,
    "whole": WholeSizeDataset,
    "image_mask": ImageMaskDataset,
    "multiscale": MultiScaleDataset,
    "supercon": SupConDataset,
}
