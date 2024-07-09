import albumentations as A
from torch.utils.data import Dataset


from .data_model import ScannerData, COSASData


class COSASdataset(Dataset):
    def __init__(self, images, masks, transform: A.Compose | None = None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
