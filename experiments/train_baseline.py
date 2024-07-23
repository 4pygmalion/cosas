import os

import mlflow
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from cosas.tracking import get_experiment
from cosas.paths import DATA_DIR
from cosas.data_model import COSASData
from cosas.datasets import DATASET_REGISTRY
from cosas.transforms import (
    train_transform,
    test_transform,
    find_representative_lab_image,
    get_lab_distribution,
)
from cosas.losses import DiceXentropy
from cosas.misc import set_seed, train_val_split, get_config
from cosas.trainer import BinaryClassifierTrainer
from cosas.tracking import TRACKING_URI, get_experiment

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)

if __name__ == "__main__":
    args = get_config()
    set_seed(42)

    if args.smp:
        model = smp.FPN(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            classes=1,
            activation=None,
        ).to(args.device)
    else:
        from cosas.networks import MODEL_REGISTRY

        model = MODEL_REGISTRY[args.model_name]().to(args.device)
    dp_model = torch.nn.DataParallel(model)

    cosas_data = COSASData(os.path.join(DATA_DIR, "task2"))
    cosas_data.load()
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = (
        train_val_split(cosas_data, train_val_test=(0.6, 0.2, 0.2))
    )
    dataset = DATASET_REGISTRY[args.dataset]

    if args.use_sn:
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

    train_dataset = dataset(
        train_images, train_masks, train_transform, device=args.device
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    if args.dataset == "pre_aug":
        dataset = DATASET_REGISTRY["image_mask"]
    val_dataset = dataset(val_images, val_masks, test_transform, device=args.device)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataset = dataset(
        test_images,
        test_masks,
        test_transform,
        device=args.device,
    )
    test_dataloder = DataLoader(test_dataset, batch_size=args.batch_size)

    dice_bce_loss = DiceXentropy()
    trainer = BinaryClassifierTrainer(
        model=dp_model,
        loss=dice_bce_loss,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        device=args.device,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = get_experiment("cosas")
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=args.run_name
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
