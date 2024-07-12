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
        self.images = images
        self.masks = masks
        self.transform = transform
        self.device = device
        self.patch_images: List[np.ndarray] = None
        self.patch_masks: List[np.ndarray] = None
        self._tesellate_patch_mask()

    def _tesellate_patch_mask(self):

        self.patch_images = []
        self.patch_masks = []
        for image, mask in zip(self.images, self.masks):
            image_tensor = ToTensor()(image)
            mask_tensor = ToTensor()(mask) * 255
            pad_image = pad_image_tensor(image_tensor)
            pad_mask = pad_image_tensor(mask_tensor)
            self.patch_images.append(tesellation(pad_image))
            self.patch_masks.append(tesellation(pad_mask))

        self.patch_images = [
            patch_tensor.permute(1, 2, 0).numpy()
            for patch_tensor in torch.concat(self.patch_images, dim=0)
        ]
        self.patch_masks = [
            patch_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
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

        # np.ndarray -> Dict[str, Tensor]
        augmented = self.transform(image=image, mask=mask)
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
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]  # (Tensor, Tensor)

        else:
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()

        mask = mask.unsqueeze(dim=0)
        patch_images, patch_masks = tesellation(image), tesellation(mask)

        return patch_images.to(self.device), patch_masks.to(self.device)
