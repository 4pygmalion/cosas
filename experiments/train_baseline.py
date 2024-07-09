# Reinhard
import os
import random

import mlflow
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage


from cosas.tracking import get_experiment

from cosas.paths import DATA_DIR
from cosas.data_model import COSASData, Scanncers
from cosas.datasets import COSASdataset
from cosas.misc import set_seed, train_val_split


if __name__ == "__main__":

    DEVICE = "cuda"
    experiment = get_experiment()
    set_seed(42)  # TODO

    cosas_data = COSASData(os.path.join(DATA_DIR, "task2"))
    cosas_data.load()

    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = (
        train_val_split(cosas_data, train_val_test=(0.6, 0.2, 0.2))
    )

    model = smp.FPN(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        classes=1,
        activation=None,
    ).to(DEVICE)

    transform = Compose(
        [
            ToPILImage(),
            Resize((256, 256)),
            ToTensor(),
        ]
    )

    breakpoint()
