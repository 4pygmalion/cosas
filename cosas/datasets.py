from typing import Tuple

import torch
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

from cosas.transforms import tesellation


class Patchdataset(Dataset):
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
        image = self.images[idx]
        mask = self.masks[idx]
        if not self.transform:
            return image, mask

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
