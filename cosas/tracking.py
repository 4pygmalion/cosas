import os
import mlflow
import numpy as np

import torch
from .misc import plot_xypred
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
    original_y: tuple,
    pred_y: torch.Tensor,
    artifact_dir: str,
):
    temp_save_path = f"{image_name}.png"
    fig, axes = plot_xypred(original_x, original_y, pred_y)
    fig.savefig(temp_save_path)
    plt.clf()
    plt.cla()
    plt.close()

    mlflow.log_artifact(temp_save_path, artifact_dir)

    os.remove(temp_save_path)
