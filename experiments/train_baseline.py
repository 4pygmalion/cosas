# Reinhard
import os
import random
import mlflow

from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

from cosas.tracking import get_experiment
from cosas.misc import set_seed
from cosas.paths import DATA_DIR
from cosas.data_model import COSASData
from cosas.datasets import COSASdataset


if __name__ == "__main__":

    experiment = get_experiment()
    set_seed(42)  # TODO

    cosas_data = COSASData(os.path.join(DATA_DIR, "task2"))
    cosas_data.load()

    train_images = []

    dataset = COSASdataset(cosas_data)
