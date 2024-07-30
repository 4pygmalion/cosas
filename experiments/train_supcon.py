"""Supervised Constrative learning

python3 experiments/train_supcon.py \
    --run_name supcon_test \
    --batch_size 512 \
    --input_size 240 \
    --model_name efficientnet-b1
"""

import os
import argparse

import mlflow
import segmentation_models_pytorch as smp
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader


from cosas.data_model import COSASData
from cosas.datasets import SupConDataset, ImageMaskDataset
from cosas.losses import SupConLoss, LOSS_REGISTRY
from cosas.misc import set_seed, get_config
from cosas.metrics import summarize_metrics
from cosas.trainer import SSLTrainer, BinaryClassifierTrainer
from cosas.tracking import TRACKING_URI, get_experiment
from cosas.transforms import CopyTransform, get_transforms
from cosas.paths import DATA_DIR

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
    parser.add_argument("--run_name", type=str, default="SuperCon", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--loss", type=str, default="SuperCon")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--n_patience", type=int, default=10, help="Number of patience epochs"
    )
    parser.add_argument("--input_size", type=int, help="Image size", required=True)
    parser.add_argument("--model_name", type=str, help="Model name")

    return parser.parse_args()


def get_encoder_transforms(input_size):
    train_transform = A.Compose(
        [
            A.RandomCrop(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=(0.9, 1.1),
                        contrast=(0.9, 1.0),
                        hue=(-0.07, 0.07),
                        saturation=(0.9, 1.1),
                    ),
                    A.ToGray(),
                ]
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            CopyTransform(p=1),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [
            A.RandomCrop(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transform, test_transform


class CustomEncoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim=384, ndim=1024):
        super().__init__()
        self.encoder = encoder
        self.avgpool = torch.nn.AvgPool2d(7)
        self.ndim = ndim
        self.hidden_dim = hidden_dim
        self.head = torch.nn.Linear(hidden_dim, ndim)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        features = self.head(x)
        return torch.nn.functional.normalize(features, dim=1)


n_hiddens = {"efficientnet-b7": 640, "efficientnet-b3": 384, "efficientnet-b1": 320}


def main(args):
    cosas_data = COSASData(os.path.join(DATA_DIR, "task2"))
    cosas_data.load()

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = get_experiment("cosas")

    summary_metrics = list()
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=args.run_name
    ):

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

            model = smp.FPN(
                encoder_name=args.model_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            ).to(args.device)
            encoder = CustomEncoder(
                model.encoder, hidden_dim=n_hiddens[args.model_name]
            ).to(args.device)
            dp_model = torch.nn.DataParallel(encoder)

            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=args.run_name + f"_Upstream{fold}",
                nested=True,
            ):
                mlflow.log_params(args.__dict__)
                mlflow.log_artifact(os.path.abspath(__file__))
                mlflow.log_artifact(os.path.join(ROOT_DIR, "cosas", "networks.py"))

                train_transform, test_transform = get_encoder_transforms(
                    args.input_size
                )
                train_dataset = SupConDataset(
                    train_images, train_masks, train_transform, device=args.device
                )
                train_dataloader = DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True
                )

                trainer = SSLTrainer(
                    model=dp_model,
                    loss=SupConLoss(),
                    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                    device=args.device,
                )
                trainer.train(train_dataloader, epochs=args.epochs)
                mlflow.pytorch.log_model(encoder, "model")
                for param in encoder.parameters():
                    param.requires_grad = False

            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=args.run_name + f"_Downstream{fold}",
                nested=True,
            ):
                train_transform, test_transform = get_transforms(args.input_size)
                train_dataloader = DataLoader(
                    ImageMaskDataset(
                        train_images, train_masks, transform=train_transform
                    ),
                    shuffle=True,
                    batch_size=args.batch_size,
                )
                val_dataloader = DataLoader(
                    ImageMaskDataset(val_images, val_masks, transform=test_transform),
                    shuffle=True,
                    batch_size=args.batch_size,
                )
                test_dataloader = DataLoader(
                    ImageMaskDataset(test_images, test_masks, transform=test_transform),
                    shuffle=True,
                    batch_size=args.batch_size,
                )

                trainer = BinaryClassifierTrainer(
                    model=model,
                    loss=LOSS_REGISTRY["mcc"](),
                    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                    device=args.device,
                )
                trainer.train(
                    train_dataloader,
                    val_dataloader,
                    epochs=args.epochs,
                    n_patience=args.n_patience,
                )
                mlflow.pytorch.log_model(model, "model")
                test_loss, test_metrics = trainer.run_epoch(
                    test_dataloader,
                    phase="test",
                    epoch=0,
                    threshold=0.5,
                    save_plot=True,
                )
                mlflow.log_metric("test_loss", test_loss.avg)
                mlflow.log_metrics(test_metrics.to_dict(prefix="test_"))
                summary_metrics.append(test_metrics.to_dict(prefix="test_"))

        mlflow.log_metrics(summarize_metrics(summary_metrics))

    return


if __name__ == "__main__":
    args = get_config()
    set_seed(42)

    main(args)
