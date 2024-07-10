from typing import Tuple

import torch
import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
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

    def __len__(self):
        return len(self.images)

    def _pad(self, image: np.ndarray, size: tuple = (224, 224)) -> Image:
        """이미지 전체를 주어진 사이즈(size)의 정수배로 만듬"""
        image_w, image_h, c = image.shape

        patch_w, patch_h = size
        pad_w = (patch_w - (image_w % patch_w)) % patch_w
        pad_h = (patch_h - (image_h % patch_h)) % patch_h

        return pad(Image.fromarray(image), (0, 0, pad_h, pad_w))

    def _tesellation(self, image: np.ndarray, size: tuple = (224, 224)) -> torch.Tensor:
        """약 1400x1400의 이미지를 패딩하여 224의 정수배로 만들어, grid로
        non overlapping 패치를 만듬

        Returns
            pathces (torch.Tensor): shape (n_patches, c, patch_h, patch_w)
        """

        patch_w, patch_h = size

        padded_image: Image = self._pad(image)
        padded_tensor: torch.Tensor = torch.from_numpy(np.array(padded_image))

        print(padded_tensor.shape)
        # (w, h, c) -> (nrow, ncol, c, w, h)
        patches = padded_tensor.unfold(0, patch_h, patch_h).unfold(1, patch_w, patch_w)
        print(patches.shape)

        # 이렇게 해야 순서대로 나오는듯
        # imgp = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute((0, 3, 4, 1, 2)).flatten(3).permute((3, 0, 1, 2))
        # imgp.shape #torch.Size([49, 3, 32, 32]
        # imgp = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute((0, 3, 4, 1, 2)).flatten(3).permute((3, 0, 1, 2))
        # imgp.shape  # torch.Size([49, 3, 32, 32]
        patches = patches.permute((1, 0, 2, 3, 4))

        return patches.contiguous().view(-1, 3, patch_h, patch_w)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]
        if not self.transform:
            return image, mask

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"].to(self.device)
        mask = augmented["mask"].to(self.device)

        return image, mask
