import os
import argparse

import mlflow
import segmentation_models_pytorch as smp
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from cosas.tracking import get_experiment
from cosas.paths import DATA_DIR
from cosas.data_model import COSASData
from cosas.datasets import DATASET_REGISTRY
from cosas.transforms import (
    RandStainNATransform,
    StainSeparationTransform,
    get_transforms,
    find_representative_lab_image,
    get_lab_distribution,
    AUG_REGISTRY,
)
from cosas.losses import LOSS_REGISTRY
from cosas.misc import set_seed, get_config
from cosas.trainer import BinaryClassifierTrainer
from cosas.tracking import TRACKING_URI, get_experiment
from cosas.metrics import summarize_metrics

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)


def stain_normalization(train_images, val_images, test_images):
    from histomicstk.preprocessing.color_normalization import reinhard
    from histomicstk.preprocessing.color_conversion import rgb_to_lab

    means, stds = get_lab_distribution(train_images)
    reference_image = find_representative_lab_image(train_images, means)
    lab_reference_image = rgb_to_lab(reference_image)
    means = lab_reference_image.mean(axis=(0, 1))
    stds = lab_reference_image.std(axis=(0, 1))
    train_images = [reinhard(image, means, stds) for image in train_images]
    val_images = [reinhard(image, means, stds) for image in val_images]
    test_images = [reinhard(image, means, stds) for image in test_images]

    return train_images, val_images, test_images


def extended_args() -> argparse.Namespace:
    args = get_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--aggregation", type=str, help="Aggregation")
    parser = parser.parse_args(namespace=args)

    return parser


if __name__ == "__main__":
    args = get_config()
    set_seed(42)

    cosas_data = COSASData(DATA_DIR, task=args.task)
    cosas_data.load()

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
            folds.split(cosas_data.images, cosas_data.domain_indices), start=1
        ):

            train_val_images = [cosas_data.images[i] for i in train_val_indices]
            train_val_masks = [cosas_data.masks[i] for i in train_val_indices]
            train_val_domains = cosas_data.domain_indices[train_val_indices]
            test_images = [cosas_data.images[i] for i in test_indices]
            test_masks = [cosas_data.masks[i] for i in test_indices]
            train_images, val_images, train_masks, val_masks = train_test_split(
                train_val_images,
                train_val_masks,
                test_size=0.2,
                random_state=args.seed,
                stratify=train_val_domains,
            )

            if args.use_sn:
                train_images, val_images, test_images = stain_normalization(
                    train_images, val_images, test_images
                )

            train_transform, test_transform = get_transforms(args.input_size)
            # Augmentation settings
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

            dataset = DATASET_REGISTRY[args.dataset]
            train_dataset = dataset(
                train_images, train_masks, train_transform, device=args.device
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )

            # VAL, TEST Dataset
            if args.dataset == "pre_aug":
                dataset = DATASET_REGISTRY["image_mask"]
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

            # TODO
            if args.smp:
                model = smp.FPN(
                    encoder_name=args.encoder_name,
                    encoder_weights=args.encoder_weights,
                    classes=1,
                    activation=None,
                ).to(args.device)
            else:
                from cosas.networks import MODEL_REGISTRY

                if args.model_name == "transunet":
                    model = MODEL_REGISTRY[args.model_name](args.input_size).to(
                        args.device
                    )
                else:
                    model = MODEL_REGISTRY[args.model_name]().to(args.device)

            # Get model params for ensemble learning (continual)
            # Model1 - MultiTaskAE - ae_efficietnet-b7_mix_fold4
            # Model2 - Segformer - segformer_iou_fold4
            model1_uri = "/vast/AI_team/mlflow_artifact/13/f7fdd3b7ef354160ab8fd4882ec2be35/artifacts/model"
            model2_uri = "/vast/AI_team/mlflow_artifact/13/c0f3923ab34842648faf50de6e4f832f/artifacts/model"

            model1_state_dict = mlflow.pytorch.load_model(model1_uri).state_dict()
            model2_state_dict = mlflow.pytorch.load_model(model2_uri).state_dict()

            model.model1.load_state_dict(model1_state_dict)
            model.model2.load_state_dict(model2_state_dict)

            dp_model = torch.nn.DataParallel(model)
            trainer = BinaryClassifierTrainer(
                model=dp_model,
                loss=LOSS_REGISTRY[args.loss](),
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
