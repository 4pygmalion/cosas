from typing import Tuple, List

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
