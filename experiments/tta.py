import argparse
from typing import Dict, List
from collections import defaultdict

import mlflow
import albumentations as A
import torch
from torchvision.transforms.functional import rotate
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from albumentations.pytorch.transforms import ToTensorV2

from cosas.losses import AELoss
from cosas.trainer import Evaluator
from cosas.data_model import COSASData
from cosas.paths import DATA_DIR
from cosas.tracking import get_child_run_ids, TRACKING_URI, get_experiment
from cosas.datasets import ImageMaskDataset
from cosas.metrics import summarize_metrics


MODEL_URI = "file:///vast/AI_team/mlflow_artifact/13/{run_id}/artifacts/model"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parent_id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    return parser.parse_args()


def tta_summary(n_trials: List[Dict[str, torch.Tensor]]):
    res = defaultdict(list)
    for trial in n_trials:
        for k, v in trial.items():
            if k in res:
                res[k].append(v)  # (n_class, W, H)
            else:
                res[k] = [v]

    return {k: torch.stack(v, dim=0).mean(dim=0) for k, v in res.items()}


@torch.no_grad()
def rotational_tta(xs, model):
    y_hats = list()
    for x in xs:
        outputs = list()
        for angle in [0, 90, 180, 270]:
            x_new = rotate(x, angle=angle)
            output: dict = model(x_new.unsqueeze(0))
            outputs.append(
                {k: rotate(tensor.squeeze(0), -angle) for k, tensor in output.items()}
            )

        y_hat = tta_summary(outputs)
        y_hats.append(y_hat)

    res = dict()
    for k in y_hats[0].keys():
        res[k] = torch.stack([y_hat[k] for y_hat in y_hats], dim=0)

    return res


if __name__ == "__main__":

    experiment = get_experiment("cosas")

    mlflow.set_tracking_uri(TRACKING_URI)
    args = get_args()

    cosas_data = COSASData(DATA_DIR, task=2)
    cosas_data.load()

    childrun_ids = get_child_run_ids(args.parent_id)[::-1]
    parent_run = mlflow.get_run(args.parent_id)
    parent_run_name = parent_run.info.run_name

    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(
        run_name=f"TTA_{parent_run_name}", experiment_id=experiment.experiment_id
    ) as run:
        summary_metrics = list()

        for fold, (train_val_indices, test_indices) in enumerate(
            folds.split(cosas_data.images, cosas_data.masks), start=1
        ):

            with mlflow.start_run(
                run_name=f"TTA_{parent_run_name}_fold_{fold}",
                nested=True,
                experiment_id=experiment.experiment_id,
            ) as childrun:
                test_images = [cosas_data.images[i] for i in test_indices]
                test_masks = [cosas_data.masks[i] for i in test_indices]

                childrun_id = childrun_ids[fold - 1]

                params: Dict[str, str] = mlflow.get_run(childrun_id).data.params
                device = params["device"]
                input_size = params["input_size"]

                model_uri = MODEL_URI.format(run_id=childrun_id)

                model = mlflow.pytorch.load_model(model_uri).eval()
                test_transform = A.Compose(
                    [
                        A.Resize(input_size, input_size),
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2(),
                    ]
                )
                test_dataset = ImageMaskDataset(
                    test_images, test_masks, test_transform, device=device
                )
                test_dataloader = DataLoader(
                    test_dataset, batch_size=32, shuffle=True
                )  # All positive, negative 때문에 shuffle
                trainer = Evaluator(
                    model=model,
                    loss=AELoss(False, alpha=1),
                    device=device,
                )
                test_loss, test_metrics = trainer.run_epoch(
                    test_dataloader,
                    threshold=0.5,
                    tta=rotational_tta,
                    save_plot=True,
                )
                mlflow.log_metric("test_loss", test_loss.avg)
                mlflow.log_metrics(test_metrics.to_dict(prefix="test_"))
                summary_metrics.append(test_metrics.to_dict(prefix="test_"))

        mlflow.log_metrics(summarize_metrics(summary_metrics))
