import os
import uuid
from typing import List

import mlflow
import numpy as np
import torch
import matplotlib.pyplot as plt

from .transforms import de_normalization
from .misc import plot_xypred, plot_patch_xypred
from .metrics import calculate_metrics


TRACKING_URI = "http://219.252.39.224:5000/"
EXP_NAME = "cosas"


def get_experiment(experiment_name=EXP_NAME):
    mlflow.set_tracking_uri(TRACKING_URI)

    client = mlflow.tracking.MlflowClient(TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        client.create_experiment(experiment_name)
        return client.get_experiment_by_name(experiment_name)

    return experiment


def get_child_run_ids(parent_run_id: str) -> List[str]:
    """MLflow 부모의 Run ID로 자식의 Run ID들을 반환함"""

    child_run_ids = []
    experiment_id = mlflow.get_run(parent_run_id).info.experiment_id
    all_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
    )

    for _, row in all_runs.iterrows():
        child_run_ids.append(row["run_id"])

    return child_run_ids


def plot_and_save(
    image_name: str,
    original_x: np.ndarray,
    original_y: np.ndarray,
    pred_y: torch.Tensor,
    artifact_dir: str,
):
    """플롯을 그리고 저장

    Args:
        image_name (str): _description_
        original_x (np.ndarray): _description_
        original_y (np.ndarray): _description_
        pred_y (torch.Tensor): (N, 224, 224, 1)
        artifact_dir (str): _description_
    """
    temp_save_path = f"{image_name}.png"
    fig, axes = plot_xypred(original_x, original_y, pred_y)
    fig.savefig(temp_save_path)
    plt.clf()
    plt.cla()
    plt.close()

    mlflow.log_artifact(temp_save_path, artifact_dir)

    os.remove(temp_save_path)


def log_patch_and_save(
    image_name: str,
    original_x: np.ndarray,
    original_y: np.ndarray,
    pred_masks: np.ndarray,
    artifact_dir: str,
):
    """플롯을 그리고 저장

    Args:
        image_name (str): _description_
        original_x (np.ndarray): _description_
        original_y (np.ndarray): _description_
        pred_y (torch.Tensor): (N, 224, 224, 1)
        artifact_dir (str): _description_
    """
    unique_id = uuid.uuid4().hex[:8]
    temp_save_path = f"{image_name}_{unique_id}.png"
    fig, axes = plot_patch_xypred(original_x, original_y, pred_masks)
    fig.savefig(temp_save_path)
    plt.clf()
    plt.cla()
    plt.close()

    mlflow.log_artifact(temp_save_path, artifact_dir)

    os.remove(temp_save_path)


def log_patch_and_save_by_batch(
    batch_xs: torch.Tensor,
    batch_ys: torch.Tensor,
    batch_confidences: torch.Tensor,
    phase: str,
    postprocess: callable = None,
    threshold=0.5,
) -> None:

    for i, (x, y, confidence) in enumerate(zip(batch_xs, batch_ys, batch_confidences)):
        image_confidence = confidence.detach().cpu().numpy()
        image_label = y.detach().cpu().numpy()

        pred_label = (image_confidence >= 0.5).astype(np.uint8)
        if postprocess is not None:
            pred_label = postprocess(pred_label).astype(np.uint8)

        instance_metrics = calculate_metrics(
            image_confidence,
            image_label,
            threshold=threshold,
            postprocess=postprocess,
        )
        dice = round(instance_metrics["dice"], 4)
        iou = round(instance_metrics["iou"], 4)

        original_x = de_normalization(normalized_image=x.permute(1, 2, 0).cpu().numpy())

        log_patch_and_save(
            image_name=f"step_{i}_dice_{dice}_iou_{iou}",
            original_x=original_x,
            original_y=image_label,
            pred_masks=pred_label,
            artifact_dir=f"{phase}_prediction",
        )

    return
