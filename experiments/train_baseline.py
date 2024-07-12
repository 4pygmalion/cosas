import os

import mlflow
import segmentation_models_pytorch as smp
import torch
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
import torch.multiprocessing as mp

from cosas.tracking import get_experiment
from cosas.paths import DATA_DIR
from cosas.data_model import COSASData
from cosas.datasets import Patchdataset, WholeSizeDataset
from cosas.misc import set_seed, train_val_split, get_config
from cosas.trainer import BinaryClassifierTrainer
from cosas.tracking import TRACKING_URI, get_experiment

if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = get_config()
    set_seed(42)

    cosas_data = COSASData(os.path.join(DATA_DIR, "task2"))
    cosas_data.load()

    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = (
        train_val_split(cosas_data, train_val_test=(0.6, 0.2, 0.2))
    )

    model = smp.FPN(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=1,
        activation=None,
    ).to(args.device)

    train_transform = A.Compose(
        [
            A.RandomCrop(height=224, width=224, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # TODO: RandomRotation90 추가
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_dataset = Patchdataset(train_images, train_masks, train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_dataset = WholeSizeDataset(val_images, val_masks, test_transform)

    n_steps = len(train_dataloader)
    trainer = BinaryClassifierTrainer(
        model,
        torch.nn.functional.binary_cross_entropy_with_logits,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = get_experiment("cosas")
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=args.run_name
    ):
        mlflow.log_params(args.__dict__)
        mlflow.log_artifact(os.path.abspath(__file__))
        trainer.train(
            train_dataloader,
            val_dataset,
            args.epochs,
            args.n_patience,
        )
        mlflow.pytorch.log_model(trainer.model, "model")

        test_dataset = WholeSizeDataset(
            test_images,
            test_masks,
            test_transform,
        )
        test_loss, test_metrics = trainer.test(
            test_dataset, phase="test", epoch=0, threshold=0.5, save_plot=True
        )
        mlflow.log_metric("test_loss", test_loss.avg)
        mlflow.log_metrics(test_metrics.to_dict(prefix="test_"))
