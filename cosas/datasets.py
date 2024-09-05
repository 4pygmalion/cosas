import random
from typing import Tuple, List
from copy import deepcopy

import cv2
import tqdm
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

from .transforms import tesellation, de_normalization
from .stain_seperation.seestaina.structure_preversing import Augmentor


class Patchdataset(Dataset):
    """한 이미지를 패치로 나눠 N개의 패치를 차원을 높여 (1, p, w, h)만든 데이터셋"""

    def __init__(
        self,
        images,
        masks,
        transform: A.Compose | None = None,
        patch_size: tuple = (512, 512),
        device: str = "cuda",
    ):
        self.images = images  # uint8
        self.masks = masks  # uint8
        self.transform = transform
        self.device = device

        # uint8
        self.patch_size = patch_size
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


class ImageClassDataset(SupConDataset):
    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        transform: A.Compose = None,
        test: bool = False,
        threshold=0.01,
        device="cuda",
        image_size=(386, 386),
        patch_sizes=[(256, 256), (386, 386), (512, 512), (768, 768)],
    ):
        self.images = deepcopy(images)
        self.masks = deepcopy(masks)
        self.labels = [
            np.array([0]) if mask.sum() == 0 else np.array([1]) for mask in self.masks
        ]
        self.transform = transform
        self.test = test
        self.threshold = threshold
        self.device = device
        self.image_size = image_size
        self.patch_sizes = patch_sizes
        if not test:
            self._preaug()

    def _maskout(self, image, mask):
        if mask.sum() == 0:
            return image, np.array([0])

        if mask.sum() == np.sum(mask.shape):
            return image, np.array([1])

        new_image = image.copy()
        new_image[np.where(mask == 1)] = 255

        return new_image, np.array([0])

    def collate_positive_mask(self):
        self.positive_patchs = list()

        patch_size = random.sample(self.patch_sizes, k=1)[0]
        crop = A.RandomCrop(*patch_size, p=1)

        for image, mask in zip(self.images, self.masks):
            aug = crop(image=image, mask=mask)
            mask_ratio = aug["mask"].sum() / np.prod(aug["mask"].shape)
            if mask_ratio > 0.05:
                self.positive_patchs.append(aug["image"])

        return

    def _paste(self, image, positive_patch) -> np.ndarray:
        new_image = image.copy()
        h, w, _ = image.shape

        pos_h, pos_w = positive_patch.shape[:2]
        start_x = np.random.randint(0, h - pos_h)
        start_y = np.random.randint(0, w - pos_w)
        new_image[start_x : start_x + pos_h, start_y : start_y + pos_w, :] = (
            positive_patch
        )

        return new_image

    def _cutpaste(self, image, mask):
        positive_patch = random.sample(self.positive_patchs, 1)[0]
        if mask.sum() == 0:
            new_image = self._paste(image, positive_patch)
            return new_image, np.array([1])

        else:
            new_image, new_label = self._maskout(image, mask)
            new_image = self._paste(new_image, positive_patch)

            return new_image, np.array([1])

    def _preaug(self, multiple=3):
        self.collate_positive_mask()

        new_images = list()
        new_labels = list()
        self.labels = [
            np.array([0]) if mask.sum() == 0 else np.array([1]) for mask in self.masks
        ]
        for _ in tqdm.tqdm(range(multiple), desc="Pre-augmentation"):
            for image, mask in zip(self.images, self.masks):
                if random.uniform(0, 1) > 0.5:
                    new_image, new_label = self._cutpaste(image, mask)
                    new_images.append(new_image)
                    new_labels.append(new_label)

                else:
                    new_image, new_label = self._maskout(image, mask)
                    new_images.append(new_image)
                    new_labels.append(new_label)

        self.images += new_images
        self.labels += new_labels

        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """한 뷰에서는 동일한 라벨이어야함."""

        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            aug = self.transform(image=image)
            image = aug["image"]

        return image, torch.from_numpy(label).float()


class StainDataset(Dataset):
    def __init__(
        self, images, masks, transform: A.Compose | None = None, device="cuda"
    ):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.device = device
        self.augmentor = Augmentor()
        self.image2density = dict()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]

        # self.transform: np.ndarray -> Dict[str, Tensor]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

            image: torch.Tensor = image
            image_size = image.shape[-2:]
            image_array = image.permute(1, 2, 0).cpu().numpy()  # float32
            original_image = (de_normalization(image_array) * 255).astype(np.uint8)
            stain_matrix = self.augmentor.get_stain_matrix(original_image)
            stain_desnity = self.augmentor.get_stain_density(
                original_image, stain_matrix
            ).T
            stain_desnity = torch.tensor(stain_desnity).reshape(2, *image_size).float()

            # stride가 negative인 경우 처리
            if isinstance(mask, torch.Tensor):
                mask = mask.clone()
            return image, mask, stain_desnity

        else:
            image = torch.from_numpy(image.copy())
            mask = torch.from_numpy(mask.copy())


class ImageMaskAuxillaryLabelDataset(Dataset):
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
                mask = mask.clone()

            label = torch.tensor([1.0]) if mask.sum() > 0 else torch.tensor([0.0])
            return image, mask, label

        else:
            image = torch.from_numpy(image.copy())
            mask = torch.from_numpy(mask.copy())
            label = torch.tensor([1.0]) if mask.sum() > 0 else torch.tensor([0.0])
            return image, mask, label


DATASET_REGISTRY = {
    "patch": Patchdataset,
    "whole": WholeSizeDataset,
    "image_cls": ImageClassDataset,
    "image_mask": ImageMaskDataset,
    "multiscale": MultiScaleDataset,
    "supercon": SupConDataset,
    "stain": StainDataset,
    "image_mask_aux": ImageMaskAuxillaryLabelDataset,
}
