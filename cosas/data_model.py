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
            self.masks.append(np.array(Image.open(mask_path)))

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
    data_dir: str
    task: int = 2

    def __post_init__(self):
        task_data_dir = os.path.join(self.data_dir, f"task{self.task}")
        self._set_domains()

        for domain in self.domains:
            subdir = os.path.join(task_data_dir, domain.value)
            setattr(self, domain.name, ScannerData(subdir))

        return

    def _set_domains(self):
        if self.task == 1:
            self.domains = Organs
        else:
            self.domains = Scanncers

        return

    def __repr__(self):
        repr = f"""COSASData(\n  data_dir={self.data_dir},\n"""

        n_images = 0
        repr_containers = list()
        for domain in self.domains:
            scanner_data = getattr(self, domain.name)
            n_images += len(scanner_data)
            repr_containers.append("  " + domain.name + "=" + scanner_data.__repr__())

        repr += "\n".join(repr_containers) + "\n"
        repr += f"  image(n={n_images})\n"
        repr += ")"

        return repr

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
