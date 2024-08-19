import os
import glob
from typing import List
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class ScannerData:
    data_dir: str
    image_paths: list = field(default_factory=list)
    mask_paths: list = field(default_factory=list)

    images: List[str] = field(default_factory=list)
    masks: List[str] = field(default_factory=list)

    def __post_init__(self):
        image_dir = os.path.join(self.data_dir, "image")
        mask_dir = os.path.join(self.data_dir, "mask")

        self.image_paths.extend(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths.extend(glob.glob(os.path.join(mask_dir, "*.png")))

        self.image_paths = sorted(self.image_paths)
        self.mask_paths = sorted(self.mask_paths)

        return

    def load(self):
        for image_path in sorted(self.image_paths):
            self.images.append(np.array(Image.open(image_path)))

        for mask_path in sorted(self.mask_paths):
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]  # dimension reduction (채널차원 동일값 중복)

            if 255 in np.unique(mask):
                mask = np.where(mask == 255, 1, mask)

            self.masks.append(mask)

        return self

    def __repr__(self) -> str:
        repr = (
            f"ScannerData(data_dir={self.data_dir}, "
            f"N images={len(self.image_paths)}, N mask={len(self.mask_paths)}"
        )
        return repr

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.images[idx], self.masks[idx]

        if isinstance(idx, slice):
            return [(self.images[i], self.masks[i]) for i in idx]


class Scanncers(Enum):
    kfbio = "kfbio-400"
    ddd = "3d-1000"
    teksqray = "teksqray-600p"


class Organs(Enum):
    colorectum = "colorectum"
    pancreas = "pancreas"
    stomach = "stomach"


@dataclass
class COSASData:
    """COSAS dataset

    Example:
        >>> from cosas.data_model import COSASData
        >>> from cosas.paths import DATA_DIR
        >>> cosas_task1 = COSASData(DATA_DIR, task=1)
        >>> cosas_task1.load()
        >>> print(cosas_task1)
        COSASData(
            data_dir=/vast/AI_team/dataset/COSAS24-TrainingSet,
            colorectum=ScannerData(data_dir=/vast/AI_team/dataset/COSAS24-TrainingSet/task1/colorectum, N images=60, N mask=60
            pancreas=ScannerData(data_dir=/vast/AI_team/dataset/COSAS24-TrainingSet/task1/pancreas, N images=60, N mask=60
            stomach=ScannerData(data_dir=/vast/AI_team/dataset/COSAS24-TrainingSet/task1/stomach, N images=60, N mask=60
            image(n=180)
        )

        >>> cosas_task2 = COSASData(DATA_DIR, task=2)
        >>> cosas_task2.load()
        >>> print(cosas_task2)
        COSASData(
            data_dir=/vast/AI_team/dataset/COSAS24-TrainingSet,
            kfbio=ScannerData(data_dir=/vast/AI_team/dataset/COSAS24-TrainingSet/task2/kfbio-400, N images=60, N mask=60
            ddd=ScannerData(data_dir=/vast/AI_team/dataset/COSAS24-TrainingSet/task2/3d-1000, N images=60, N mask=60
            teksqray=ScannerData(data_dir=/vast/AI_team/dataset/COSAS24-TrainingSet/task2/teksqray-600p, N images=60, N mask=60
            image(n=180)
        )
    """

    data_dir: str
    task: int = 2

    def __post_init__(self):
        task_data_dir = os.path.join(self.data_dir, f"task{self.task}")
        self.domains = Organs if self.task == 1 else Scanncers

        for domain in self.domains:
            subdir = os.path.join(task_data_dir, domain.value)
            setattr(self, domain.name, ScannerData(subdir))

        return

    def __repr__(self):
        output = f"""COSASData(\n  data_dir={self.data_dir},\n"""

        n_images = 0
        subdata_repr = list()
        for domain in self.domains:
            scanner_data = getattr(self, domain.name)
            n_images += len(scanner_data)
            subdata_repr.append(f"  {domain.name} = {scanner_data.__repr__()}")

        output += "\n".join(subdata_repr) + "\n"
        output += f"  image(n={n_images})\n"
        output += ")"

        return output

    def load(self):
        for domain in self.domains:
            getattr(self, domain.name).load()

        return self

    @property
    def images(self) -> List[np.ndarray]:
        images = list()
        for domain in self.domains:
            domain_data: ScannerData = getattr(self, domain.name)
            images.extend(domain_data.images)

        return images

    @property
    def masks(self) -> List[np.ndarray]:
        masks = list()
        for domain in self.domains:
            domain_data: ScannerData = getattr(self, domain.name)
            masks.extend(domain_data.masks)

        return masks
