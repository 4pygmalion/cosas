import random
from typing import Tuple, List


import tqdm
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor

from cosas.transforms import tesellation, pad_image_tensor


class Patchdataset(Dataset):
    def __init__(
        self, images, masks, transform: A.Compose | None = None, device: str = "cuda"
    ):
        self.images = images  # uint8
        self.masks = masks  # uint8
        self.transform = transform
        self.device = device

        # uint8
        self.patch_images: List[np.ndarray] = None
        self.patch_masks: List[np.ndarray] = None
        self._tesellate_patch_mask()

    def _tesellate_patch_mask(self):

        self.patch_images = list()
        self.patch_masks = list()
        for image, mask in zip(self.images, self.masks):
            image_tensor = ToTensor()(image)
            mask_tensor = ToTensor()(mask)  # [0, 1] normalized
            pad_image = pad_image_tensor(image_tensor)
            pad_mask = pad_image_tensor(mask_tensor)
            self.patch_images.append(tesellation(pad_image))
            self.patch_masks.append(tesellation(pad_mask))

        # (n, c, h, w), uint8
        self.patch_images = [
            (patch_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            for patch_tensor in torch.concat(self.patch_images, dim=0)
        ]
        self.patch_masks = [
            (patch_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            for patch_tensor in torch.concat(self.patch_masks, dim=0)
        ]

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


class PreAugDataset(Dataset):
    def __init__(
        self, images, masks, transform: A.Compose | None = None, device="cuda"
    ):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.device = device
        self._pre_augmentation()

    def _sample_random_size(self, image_shape):
        w, h, _ = image_shape

        new_w = int(random.uniform(w / 2, w))
        new_h = int(random.uniform(h / 2, h))
        while new_w >= w or new_h >= h or new_h <= 0 or new_w <= 0:
            new_w = int(random.uniform(w / 2, w))
            new_h = int(random.uniform(h / 2, h))

        return new_w, new_h

    def _patch_to_original(self, patch, original_shape):

        new_image = np.zeros(original_shape, dtype=np.uint8)
        h, w, _ = new_image.shape
        ch, cw, _ = patch.shape

        # 랜덤 위치 선정
        x = np.random.randint(0, w - cw)
        y = np.random.randint(0, h - ch)

        # 배경에 패치 붙이기
        new_image[y : y + ch, x : x + cw, :] = patch

        return new_image

    def _pre_transform(self, image, mask):
        negative_image = image.copy()
        negative_image[np.where(mask == 1)] = 0

        new_size = self._sample_random_size(image.shape)
        transform = A.Compose([A.RandomCrop(*new_size), A.SafeRotate()])
        aug = transform(image=negative_image, mask=mask)
        cropped_image = aug["image"]
        while len(np.unique(cropped_image)) == 1:
            aug = transform(image=negative_image, mask=mask)
            cropped_image = aug["image"]
        patched_image = self._patch_to_original(cropped_image, image.shape)

        return patched_image, np.zeros_like(mask, dtype=np.uint8)

    def _pre_augmentation(self, multiple: int = 1):

        aug_images = list()
        aug_masks = list()
        for iter in range(multiple):
            for image, mask in tqdm.tqdm(
                zip(self.images, self.masks), desc="Pre-augmentation"
            ):
                if len(np.unique(mask)) == 1 and np.unique(mask) == np.array([1]):
                    continue

                patched_image, patched_mask = self._pre_transform(image, mask)
                if patched_image.sum() == 0:
                    continue

                aug_images.append(patched_image)
                aug_masks.append(patched_mask)

        self.images = self.images + aug_images
        self.masks = self.masks + aug_masks

        return

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


class SupConDataset(Dataset):
    def __init__(
        self, images, masks, transform: A.Compose, threshold=0.01, device="cuda"
    ):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.threshold = threshold
        self.device = device
        self._preaug()

    def _preaug(self, n=16):
        self.crop = A.RandomCrop(360, 360, p=1)

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
    "pre_aug": PreAugDataset,
}
