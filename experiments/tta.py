import argparse
from typing import Tuple, List

import numpy as np
import mlflow
import albumentations as A
import torch
from PIL import Image
from progress.bar import Bar
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from albumentations.pytorch.transforms import ToTensorV2

from cosas.metrics import AverageMeter, Metrics, calculate_metrics
from cosas.losses import AELoss
from cosas.trainer import BinaryClassifierTrainer
from cosas.data_model import COSASData
from cosas.paths import DATA_DIR
from cosas.tracking import (
    get_child_run_ids,
    TRACKING_URI,
    get_experiment,
    log_patch_and_save,
)
from cosas.datasets import ImageMaskDataset
from cosas.metrics import summarize_metrics
from cosas.misc import rotational_tta, rotational_tta_dict
from cosas.normalization import SPCNNormalizer
from cosas.transforms import POSTPROCESS_REGISTRY
import os
import os


MODEL_URI = "file:///vast/AI_team/mlflow_artifact/13/{run_id}/artifacts/model"
mlflow.set_tracking_uri(TRACKING_URI)
MLFLOW_EXP = get_experiment("cosas")


class Evaluator(BinaryClassifierTrainer):
    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        device: str = "cuda",
    ):
        self.model = model
        self.loss = loss
        self.device = device

    @torch.no_grad()
    def run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        threshold: float = 0.5,
        save_plot: bool = False,
        tta: callable = None,
        postprocess: callable = None,
        model_return_dict: bool = False,
    ) -> Tuple[AverageMeter, Metrics]:
        """1회 Epoch을 각 페이즈(train, validation)에 따라서 학습하거나 손실값을
        반환함.

        Note:
            - 1 epoch = Dataset의 전체를 학습한경우
            - 1 step = epoch을 하기위해 더 작은 단위(batch)로 학습할 떄의 단위

        Args:
            phase (str): training or validation
            epoch (int): epoch
            dataloader (torch.utils.data.DataLoader): dataset (train or validation)

        Returns:
            Tuple: loss, accuracy, top_k_recall
        """

        self.model.eval()

        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        epoch_metrics = Metrics()
        loss_meter = AverageMeter("loss")
        i = 0
        for step, batch in enumerate(dataloader):
            xs, ys = batch
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            outputs = tta(xs, self.model) if tta else self.model(xs)

            if model_return_dict:
                recon_x = outputs["recon"]
                logits = outputs["mask"]
                vector = outputs["vector"]
                density = outputs["density"]
            else:
                logits = outputs
            logits = logits.view(ys.shape)

            images_confidences = torch.sigmoid(logits)
            flat_confidence = images_confidences.flatten().detach().cpu().numpy()
            ground_truths: torch.Tensor = ys.flatten().detach().cpu().numpy()

            epoch_metrics.update(
                calculate_metrics(
                    flat_confidence,
                    ground_truths,
                    threshold=threshold,
                    postprocess=postprocess,
                )
            )

            if save_plot:
                for i, (x, y, confidences) in enumerate(
                    zip(xs, ys, images_confidences)
                ):
                    image_confidences = confidences.detach().cpu().numpy()
                    image_lebels = y.detach().cpu().numpy()
                    instance_metrics = calculate_metrics(
                        image_confidences.ravel(),
                        image_lebels.ravel(),
                        threshold=threshold,
                        postprocess=postprocess,
                    )
                    dice = round(instance_metrics["dice"], 4)
                    iou = round(instance_metrics["iou"], 4)

                    mean = [0.485, 0.456, 0.406]
                    sd = [0.229, 0.224, 0.225]
                    original_x = ToPILImage()(
                        x.detach().cpu() * torch.tensor(sd)[:, None, None]
                        + torch.tensor(mean)[:, None, None]
                    )
                    log_patch_and_save(
                        image_name=f"step_{i}_dice_{dice}_iou_{iou}",
                        original_x=np.array(original_x),
                        original_y=image_lebels,
                        pred_masks=image_confidences >= 0.5,
                        artifact_dir="test_prediction",
                    )
                    i += 1

            bar.suffix = self.make_bar_sentence(
                phase="test",
                epoch=0,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=loss_meter.avg,
                metrics=epoch_metrics,
            )
            bar.next()

        bar.finish()

        return (loss_meter, epoch_metrics)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parent_id", type=str, required=True)
    parser.add_argument("-t", "--task", type=int, required=True, help="Task number")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--use_sn", action="store_true", help="Use stain normalization")
    parser.add_argument("--use_tta", help="Use rotational tta", action="store_true")
    parser.add_argument(
        "--postprocess",
        type=str,
        choices=list(POSTPROCESS_REGISTRY.keys()),
        default=None,
        help="Use postprocessing",
    )
    parser.add_argument(
        "--model_return_dict",
        help="Model return dict type",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


def load_data(task: int = 2):
    cosas_data = COSASData(DATA_DIR, task=task)
    cosas_data.load()
    return cosas_data


def stain_normalization(images: List[np.ndarray]):
    normalizer = SPCNNormalizer()

    target_image = Image.open(
        "/vast/AI_team/dataset/COSAS24-TrainingSet/task2/3d-1000/image/db4b0298b346.png"
    ).resize((512, 512))

    normalizer.fit(np.array(target_image))

    new_images = list()
    for image in images:
        new_images.append(normalizer.transform(image))
    return new_images


def prepare_test_dataloader(test_images, test_masks, input_size, batch_size, device):
    test_transform = A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    test_dataset = ImageMaskDataset(
        test_images, test_masks, test_transform, device=device
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def process_fold(
    val_dataloader,
    test_dataloader,
    fold,
    evaluator: Evaluator,
    parent_run_name,
    postprocess,
    model_return_dict,
    tta_fn: callable = None,
):
    with mlflow.start_run(
        run_name=f"TTA_{parent_run_name}_fold_{fold}",
        nested=True,
        experiment_id=MLFLOW_EXP.experiment_id,
    ):

        val_loss, val_metrics = evaluator.run_epoch(
            val_dataloader,
            threshold=0.5,
            tta=tta_fn,
            postprocess=postprocess,
            save_plot=True,
            model_return_dict=model_return_dict,
        )
        mlflow.log_metric("val_loss", val_loss.avg)
        mlflow.log_metrics(val_metrics.to_dict(prefix="val_"))

        test_loss, test_metrics = evaluator.run_epoch(
            test_dataloader,
            threshold=0.5,
            tta=tta_fn,
            postprocess=postprocess,
            save_plot=True,
            model_return_dict=model_return_dict,
        )

        mlflow.log_metric("test_loss", test_loss.avg)
        mlflow.log_metrics(test_metrics.to_dict(prefix="test_"))

    return test_metrics.to_dict(prefix="test_")


def main():
    args = get_args()

    cosas_data: COSASData = load_data(task=args.task)

    parent_run = mlflow.get_run(args.parent_id)
    parent_run_name = parent_run.info.run_name
    childrun_ids = get_child_run_ids(args.parent_id)[::-1]  # fold ascending order

    summary_metrics = []
    folds = StratifiedKFold(n_splits=4, random_state=args.seed, shuffle=True)
    with mlflow.start_run(
        run_name=f"TTA_{parent_run_name}", experiment_id=MLFLOW_EXP.experiment_id
    ) as run:
        mlflow.log_params(args.__dict__)

        for fold, (train_val_indices, test_indices) in enumerate(
            folds.split(cosas_data.images, cosas_data.domain_indices), start=1
        ):
            child_run_id = childrun_ids[fold - 1]
            child_run = mlflow.get_run(child_run_id)
            params = child_run.data.params

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
                val_images = stain_normalization(val_images)
                test_images = stain_normalization(test_images)
            val_dataloader = prepare_test_dataloader(
                val_images,
                val_masks,
                int(params["input_size"]),
                args.batch_size,
                args.device,
            )
            test_dataloader = prepare_test_dataloader(
                test_images,
                test_masks,
                int(params["input_size"]),
                args.batch_size,
                args.device,
            )

            model_uri = MODEL_URI.format(run_id=child_run_id)
            model = mlflow.pytorch.load_model(model_uri).eval()
            evaluator = Evaluator(
                model=model, loss=AELoss(False, alpha=1), device=args.device
            )

            if not args.use_tta:
                tta_fn = None
            elif args.model_return_dict:
                tta_fn = rotational_tta_dict
            else:
                tta_fn = rotational_tta

            if args.postprocess:
                postprocess = POSTPROCESS_REGISTRY[args.postprocess]

            metrics = process_fold(
                val_dataloader,
                test_dataloader,
                fold,
                evaluator,
                parent_run_name,
                postprocess=postprocess,
                model_return_dict=args.model_return_dict,
                tta_fn=tta_fn,
            )
            summary_metrics.append(metrics)

        mlflow.log_metrics(summarize_metrics(summary_metrics))


if __name__ == "__main__":
    main()
