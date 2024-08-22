import argparse
from typing import Tuple

import numpy as np
import mlflow
import albumentations as A
import torch
from progress.bar import Bar
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
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
from cosas.misc import rotational_tta


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

            recon_x = outputs["recon"]
            logits = outputs["mask"]
            vector = outputs["vector"]
            density = outputs["density"]
            logits = logits.view(ys.shape)
            loss = self.loss(recon_x, xs, logits, ys.float(), vector, density)

            # metric
            loss_meter.update(loss.item(), len(ys))

            images_confidences = torch.sigmoid(logits)
            flat_confidence = images_confidences.flatten().detach().cpu().numpy()
            ground_truths: torch.Tensor = ys.flatten().detach().cpu().numpy()

            epoch_metrics.update(
                calculate_metrics(
                    flat_confidence,
                    ground_truths,
                    threshold=threshold,
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
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def load_data(task: int = 2):
    cosas_data = COSASData(DATA_DIR, task=task)
    cosas_data.load()
    return cosas_data


def prepare_test_dataloader(test_images, test_masks, input_size, device):
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
    return DataLoader(test_dataset, batch_size=128, shuffle=False)


def process_fold(
    test_dataloader,
    fold,
    evaluator,
    parent_run_name,
    tta_fn: callable = None,
):
    with mlflow.start_run(
        run_name=f"TTA_{parent_run_name}_fold_{fold}",
        nested=True,
        experiment_id=MLFLOW_EXP.experiment_id,
    ):

        test_loss, test_metrics = evaluator.run_epoch(
            test_dataloader, threshold=0.5, tta=tta_fn, save_plot=True
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
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(
        run_name=f"TTA_{parent_run_name}", experiment_id=MLFLOW_EXP.experiment_id
    ) as run:
        for fold, (_, test_indices) in enumerate(
            folds.split(cosas_data.images, cosas_data.masks), start=1
        ):
            child_run_id = childrun_ids[fold - 1]
            child_run = mlflow.get_run(child_run_id)
            params = child_run.data.params

            test_images = [cosas_data.images[i] for i in test_indices]
            test_masks = [cosas_data.masks[i] for i in test_indices]
            test_dataloader = prepare_test_dataloader(
                test_images, test_masks, int(params["input_size"]), args.device
            )

            model_uri = MODEL_URI.format(run_id=child_run_id)
            model = mlflow.pytorch.load_model(model_uri).eval()
            evaluator = Evaluator(
                model=model, loss=AELoss(False, alpha=1), device=args.device
            )
            metrics = process_fold(
                test_dataloader,
                fold,
                evaluator,
                parent_run_name,
                tta_fn=rotational_tta,
            )
            summary_metrics.append(metrics)

        mlflow.log_metrics(summarize_metrics(summary_metrics))


if __name__ == "__main__":
    main()
