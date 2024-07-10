from typing import Tuple

import torch
import albumentations as A
from torch.utils.data import Dataset

from .data_model import ScannerData, COSASData


class COSASdataset(Dataset):
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
