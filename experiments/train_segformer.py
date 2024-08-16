import os

import mlflow
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from cosas.tracking import get_experiment
from cosas.paths import DATA_DIR
from cosas.data_model import COSASData
from cosas.datasets import DATASET_REGISTRY
from cosas.transforms import get_transforms
from cosas.losses import LOSS_REGISTRY
from cosas.networks import MODEL_REGISTRY
from cosas.misc import set_seed, get_config, CosineAnnealingWarmUpRestarts
from cosas.trainer import BinaryClassifierTrainer
from cosas.tracking import TRACKING_URI, get_experiment
from cosas.metrics import summarize_metrics
from transformers import SegformerModel

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)


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

            segformer = MODEL_REGISTRY["segformer"]().to(args.device)
            dp_model = torch.nn.DataParallel(segformer)
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": segformer.model.decode_head.parameters(),
                        "lr": args.lr * 10.0,
                    },
                    {
                        "params": [
                            param
                            for name, param in segformer.named_parameters()
                            if "decode_head" not in name
                        ]
                    },
                ],
                lr=args.lr,
            )
            trainer = BinaryClassifierTrainer(
                model=dp_model,
                loss=LOSS_REGISTRY[args.loss](),
                optimizer=optimizer,
                scheduler=CosineAnnealingWarmUpRestarts(
                    optimizer, T_0=150, gamma=0.8, eta_max=1e-5
                ),
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
                mlflow.pytorch.log_model(segformer, "model")

                test_loss, test_metrics = trainer.run_epoch(
                    test_dataloder, phase="test", epoch=0, threshold=0.5, save_plot=True
                )
                mlflow.log_metric("test_loss", test_loss.avg)
                mlflow.log_metrics(test_metrics.to_dict(prefix="test_"))
                summary_metrics.append(test_metrics.to_dict(prefix="test_"))

        mlflow.log_metrics(summarize_metrics(summary_metrics))
