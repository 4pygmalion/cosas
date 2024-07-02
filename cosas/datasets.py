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

    image: List[str] = field(default_factory=list)
    mask: List[str] = field(default_factory=list)

    def __post_init__(self):
        image_dir = os.path.join(self.data_dir, "image")
        mask_dir = os.path.join(self.data_dir, "mask")

        self.image_paths.extend(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths.extend(glob.glob(os.path.join(mask_dir, "*.png")))

        return

    def load(self):
        for image_path in sorted(self.image_paths):
            self.image.append(np.array(Image.open(image_path)))

        for mask_path in sorted(self.mask_paths):
            self.mask.append(np.array(Image.open(mask_path)))

        return self

    def __repr__(self) -> str:
        repr = (
            f"ScannerData(data_dir={self.data_dir}, "
            f"N images={len(self.image_paths)}, N mask={len(self.mask_paths)}"
        )
        return repr

    def __len__(self) -> int:
        return len(self.image_paths)


class Scanncers(Enum):
    kfbio = "kfbio-400"
    ddd = "3d-1000"
    teksqray = "teksqray-600p"


@dataclass
class COSASData:
    data_dir: str

    def __post_init__(self):
        for scanner in Scanncers:
            scanner_dir = os.path.join(self.data_dir, scanner.value)
            setattr(self, scanner.name, ScannerData(scanner_dir))

        return

    def __repr__(self):
        repr = f"""COSASData(\n  data_dir={self.data_dir},\n"""

        n_images = 0
        scanner_repr = list()
        for scanner in Scanncers:
            scanner_data = getattr(self, scanner.name)
            n_images += len(scanner_data)
            scanner_repr.append("  " + scanner.name + "=" + scanner_data.__repr__())

        repr += "\n".join(scanner_repr) + "\n"
        repr += f"  image(n={n_images})\n"
        repr += ")"

        return repr

    def load(self):
        for scanner in Scanncers:
            getattr(self, scanner.name).load()

        return self

    @property
    def image(self) -> List[np.ndarray]:
        images = list()
        for scanner in Scanncers:
            scanner_data: ScannerData = getattr(self, scanner.name)
            images.extend(scanner_data.image)

        return images
