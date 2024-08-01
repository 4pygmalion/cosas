import os
import uuid
import mlflow
import numpy as np

import torch
from .misc import plot_xypred, plot_patch_xypred
import matplotlib.pyplot as plt

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
