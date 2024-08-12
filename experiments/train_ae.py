import os
import argparse

import mlflow
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from cosas.tracking import get_experiment
from cosas.paths import DATA_DIR
from cosas.networks import MultiTaskAE, MultiTaskTransAE
from cosas.data_model import COSASData
from cosas.datasets import DATASET_REGISTRY
from cosas.transforms import CopyTransform
from cosas.losses import AELoss
from cosas.misc import set_seed, get_config
from cosas.trainer import AETrainer
from cosas.tracking import TRACKING_URI, get_experiment
from cosas.metrics import summarize_metrics

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)


def get_config() -> argparse.ArgumentParser:
    """
    Note:
        num_workers 필요없음. 이미 메모리에 다 올려서 필요없는듯
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run_name", type=str, default="baseline", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--dataset",
        type=str,
        default="image_mask",
        choices=list(DATASET_REGISTRY.keys()),
        required=False,
    )
    parser.add_argument("--loss", type=str, default="multi-task", required=False)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--n_patience", type=int, default=10, help="Number of patience epochs"
    )
    parser.add_argument("--input_size", type=int, help="Image size", required=True)
    parser.add_argument(
        "--model_name", type=str, help="Model name", default="autoencoder"
    )
    arch_choices = [
        "Unet",
        "UnetPlusPlus",
        "MAnet",
        "Linknet",
        "FPN",
        "PSPNet",
        "DeepLabV3",
        "DeepLabV3Plus",
        "PAN",
        "TransUNet",
    ]
    parser.add_argument("--architecture", type=str, required=True, choices=arch_choices)
    parser.add_argument("--encoder_name", type=str, required=True)

    parser.add_argument("--use_sparisty_loss", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1.0)

    return parser.parse_args()


def get_transforms(input_size):
    train_transform = A.Compose(
        [
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            CopyTransform(p=1),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transform, test_transform


if __name__ == "__main__":
    args = get_config()
    set_seed(42)

    cosas_data = COSASData(os.path.join(DATA_DIR, "task2"))
    cosas_data.load()

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = get_experiment("cosas")

    summary_metrics = list()
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=args.run_name
    ) as run:
        folds = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_val_indices, test_indices) in enumerate(
            folds.split(cosas_data.images, cosas_data.masks), start=1
        ):
            train_val_images = [cosas_data.images[i] for i in train_val_indices]
            train_val_masks = [cosas_data.masks[i] for i in train_val_indices]
            test_images = [cosas_data.images[i] for i in test_indices]
            test_masks = [cosas_data.masks[i] for i in test_indices]

            train_images, val_images, train_masks, val_masks = train_test_split(
                train_val_images, train_val_masks, test_size=0.2, random_state=args.seed
            )
            dataset = DATASET_REGISTRY[args.dataset]

            train_transform, test_transform = get_transforms(args.input_size)
            train_dataset = dataset(
                train_images, train_masks, train_transform, device=args.device
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )

            # VAL, TEST Dataset
            val_dataset = dataset(
                val_images, val_masks, test_transform, device=args.device
            )
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
            test_dataset = dataset(
                test_images,
                test_masks,
                test_transform,
                device=args.device,
            )
            test_dataloder = DataLoader(test_dataset, batch_size=args.batch_size)

            if args.architecture != "TransUNet":
                model = MultiTaskAE(
                    architecture=args.architecture,
                    encoder_name=args.encoder_name,
                    input_size=(args.input_size, args.input_size),
                ).to(args.device)
            else:
                model = MultiTaskTransAE(
                    architecture=args.architecture,
                    encoder_name="vit",
                    input_size=(args.input_size, args.input_size),
                ).to(args.device)

            dp_model = torch.nn.DataParallel(model)
            trainer = AETrainer(
                model=dp_model,
                loss=AELoss(args.use_sparisty_loss, alpha=args.alpha),
                optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                device=args.device,
            )

            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=args.run_name + f"_fold{fold}",
                nested=True,
            ):
                mlflow.log_params(args.__dict__)
                mlflow.log_artifact(os.path.abspath(__file__))
                mlflow.log_artifact(os.path.join(ROOT_DIR, "cosas", "networks.py"))

                trainer.train(
                    train_dataloader,
                    val_dataloader,
                    epochs=args.epochs,
                    n_patience=args.n_patience,
                )
                mlflow.pytorch.log_model(model, "model")

                test_loss, test_metrics = trainer.run_epoch(
                    test_dataloder, phase="test", epoch=0, threshold=0.5, save_plot=True
                )
                mlflow.log_metric("test_loss", test_loss.avg)
                mlflow.log_metrics(test_metrics.to_dict(prefix="test_"))
                summary_metrics.append(test_metrics.to_dict(prefix="test_"))

        mlflow.log_metrics(summarize_metrics(summary_metrics))
