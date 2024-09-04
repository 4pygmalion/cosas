import os
import argparse

import torchvision
import mlflow
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader


from cosas.tracking import get_experiment
from cosas.paths import DATA_DIR
from cosas.networks import MultiTaskAE, MODEL_REGISTRY
from cosas.data_model import COSASData
from cosas.datasets import DATASET_REGISTRY, ImageClassDataset
from cosas.transforms import (
    CopyTransform,
    AUG_REGISTRY,
    RandStainNATransform,
    StainSeparationTransform,
    AUG_REGISTRY,
)
from cosas.losses import LOSS_REGISTRY, ReconMCCLoss
from cosas.misc import set_seed, get_config
from cosas.trainer import AETrainer, BinaryClassifierTrainer
from cosas.tracking import TRACKING_URI, get_experiment
from cosas.metrics import summarize_metrics
from cosas.normalization import find_median_lab_image, SPCNNormalizer

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
    parser.add_argument("--update_step", type=int, default=1, help="Update step")
    parser.add_argument(
        "--dataset",
        type=str,
        default="image_mask",
        choices=list(DATASET_REGISTRY.keys()),
        required=False,
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="recon_mcc",
        choices=list(LOSS_REGISTRY.keys()),
        required=False,
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--n_patience", type=int, default=10, help="Number of patience epochs"
    )
    parser.add_argument("--input_size", type=int, help="Image size", required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name",
        required=False,
        choices=list(MODEL_REGISTRY.keys()),
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
    parser.add_argument(
        "--architecture", type=str, required=False, choices=arch_choices
    )
    parser.add_argument("--encoder_name", type=str, required=False)

    parser.add_argument("--use_sparisty_loss", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--sa", choices=list(AUG_REGISTRY.keys()), help="Use stain augmentation"
    )
    parser.add_argument("--use_task1", action="store_true", default=False)

    return parser.parse_args()


def get_transforms(
    input_size, randstainna_transform=None, stain_separation_transform=None
):
    train_transform = [
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        CopyTransform(p=1),
        ToTensorV2(),
    ]
    if randstainna_transform and stain_separation_transform:
        train_transform.insert(
            -3, A.OneOf([randstainna_transform, stain_separation_transform])
        )
    elif randstainna_transform is not None:
        train_transform.insert(-3, randstainna_transform)
    elif stain_separation_transform is not None:
        train_transform.insert(-3, stain_separation_transform)

    test_transform = [
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    return A.Compose(train_transform), A.Compose(test_transform)


if __name__ == "__main__":
    args = get_config()
    set_seed(42)

    cosas_data2 = COSASData(DATA_DIR, task=2)
    cosas_data2.load()

    if args.use_task1:
        cosas_data1 = COSASData(DATA_DIR, task=1)
        cosas_data1.load()

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = get_experiment("cosas")

    summary_metrics = list()
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=args.run_name
    ) as run:
        folds = StratifiedKFold(n_splits=4, random_state=args.seed, shuffle=True)
        mlflow.log_params(args.__dict__)
        mlflow.log_artifacts(os.path.join(ROOT_DIR, "cosas"), artifact_path="cosas")
        mlflow.log_artifact(os.path.abspath(__file__))

        for fold, (train_val_indices, test_indices) in enumerate(
            folds.split(cosas_data2.images, cosas_data2.domain_indices), start=1
        ):

            train_val_images = [cosas_data2.images[i] for i in train_val_indices]
            train_val_masks = [cosas_data2.masks[i] for i in train_val_indices]
            train_val_domains = cosas_data2.domain_indices[train_val_indices]
            test_images = [cosas_data2.images[i] for i in test_indices]
            test_masks = [cosas_data2.masks[i] for i in test_indices]
            train_images, val_images, train_masks, val_masks = train_test_split(
                train_val_images,
                train_val_masks,
                test_size=0.2,
                random_state=args.seed,
                stratify=train_val_domains,
            )

            # Append COSAS Task1 data
            train_images += cosas_data1.images if args.use_task1 else list()
            train_masks += cosas_data1.masks if args.use_task1 else list()

            # Train dataset
            dataset = ImageClassDataset
            train_transform, test_transform = get_transforms(args.input_size)
            if args.sa == "albu_randstainna":
                randstainna_transform = RandStainNATransform()
                randstainna_transform.fit(train_images)

                train_transform, test_transform = get_transforms(
                    args.input_size,
                    randstainna_transform=randstainna_transform,
                    stain_separation_transform=None,
                )
            elif args.sa == "albu_stain_separation":
                stain_separation_transform = StainSeparationTransform()
                train_transform, test_transform = get_transforms(
                    args.input_size,
                    randstainna_transform=None,
                    stain_separation_transform=stain_separation_transform,
                )
            elif args.sa == "albu_mix":
                randstainna_transform = RandStainNATransform()
                randstainna_transform.fit(train_images)

                stain_separation_transform = StainSeparationTransform()
                train_transform, test_transform = get_transforms(
                    args.input_size,
                    randstainna_transform=randstainna_transform,
                    stain_separation_transform=stain_separation_transform,
                )
            elif args.sa:
                aug_fn = AUG_REGISTRY[args.sa]
                train_images, train_masks = aug_fn(train_images, train_masks)

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
                test_images, test_masks, test_transform, device=args.device, test=True
            )
            test_dataloder = DataLoader(test_dataset, batch_size=args.batch_size)

            # MODEL
            model = MultiTaskAE(
                architecture="Unet",
                encoder_name="efficientnet-b7",
                input_size=(args.input_size, args.input_size),
            )

            class EncoderClassifier(torch.nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.encoder = encoder
                    self.classifier = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False
                        ),
                        torch.nn.BatchNorm2d(
                            2560,
                            eps=0.001,
                            momentum=0.01,
                            affine=True,
                            track_running_stats=True,
                        ),
                        torch.nn.SiLU(inplace=True),
                        torch.nn.AdaptiveAvgPool2d(output_size=1),
                        torch.nn.Flatten(),
                        torch.nn.Linear(in_features=2560, out_features=1, bias=True),
                    )

                def forward(self, x):
                    z = self.encoder(x)[-1]
                    return self.classifier(z)

            model = model.to(args.device)
            encoder = EncoderClassifier(model.encoder).to(args.device)
            dp_encoder_model = torch.nn.DataParallel(encoder)

            # Loss
            loss = torch.nn.BCEWithLogitsLoss()
            trainer = BinaryClassifierTrainer(
                model=dp_encoder_model,
                loss=loss,
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
                    epochs=20,
                    n_patience=args.n_patience,
                    update_step=args.update_step,
                )
                mlflow.pytorch.log_model(model, "model")

            # Loss
            dp_model = torch.nn.DataParallel(model)
            loss = ReconMCCLoss(alpha=args.alpha)
            trainer = AETrainer(
                model=dp_model,
                loss=loss,
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
                    update_step=args.update_step,
                )
                mlflow.pytorch.log_model(model, "model")

                test_loss, test_metrics = trainer.run_epoch(
                    test_dataloder, phase="test", epoch=0, threshold=0.5, save_plot=True
                )
                mlflow.log_metric("test_loss", test_loss.avg)
                mlflow.log_metrics(test_metrics.to_dict(prefix="test_"))
                summary_metrics.append(test_metrics.to_dict(prefix="test_"))

        mlflow.log_metrics(summarize_metrics(summary_metrics))
