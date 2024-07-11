from typing import Tuple

import torch
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pad


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

    def _pad(
        self, image_tensor: torch.Tensor, size: tuple = (224, 224)
    ) -> torch.Tensor:
        """이미지 전체를 주어진 사이즈(size)의 정수배로 만듬

        Params:
            image_tensor (torch.Tensor): shape (C, W, H)
        """

        c, image_w, image_h = image_tensor.size()

        patch_w, patch_h = size
        pad_w = (patch_w - (image_w % patch_w)) % patch_w
        pad_h = (patch_h - (image_h % patch_h)) % patch_h

        return pad(image_tensor, (0, 0, pad_h, pad_w))

    def _tesellation(
        self, image_tensor: torch.Tensor, size: tuple = (224, 224)
    ) -> torch.Tensor:
        """약 1400x1400의 이미지를 패딩하여 224의 정수배로 만들어, grid로
        non overlapping 패치를 만듬

        Returns
            pathces (torch.Tensor): shape (n_patches, c, patch_h, patch_w)
        """

        patch_w, patch_h = size

        padded_tensor: torch.Tensor = self._pad(image_tensor)

        return (
            padded_tensor.unfold(1, patch_h, patch_h)
            .unfold(2, patch_w, patch_w)  # (C, row, col, w, h)
            .permute((0, 3, 4, 1, 2))  # (C, w, h, row, col)
            .flatten(3)  # (C, w, h, N)
            .permute((3, 0, 1, 2))  # (N, C, W, H)
        )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = (augmented["image"], augmented["mask"])  # (Tensor, Tensor)

        else:
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()

        mask = mask.unsqueeze(dim=0)
        patch_images, patch_masks = self._tesellation(image), self._tesellation(mask)

        return patch_images.to(self.device), patch_masks.to(self.device)
