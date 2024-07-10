import math
import logging
from typing import Tuple, Dict
from abc import ABC, abstractmethod

import mlflow
from scipy.special import softmax
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from progress.bar import Bar
from sklearn.metrics import roc_auc_score

from .metrics import Metrics, AverageMeter, calculate_metrics


class BaseTrainer(ABC):
    @abstractmethod
    def make_bar_sentence(self):
        pass

    @abstractmethod
    def run_epoch(self):
        pass

    def get_accuracy(
        self, logit: torch.Tensor, labels: torch.Tensor, threshold: int = 0.5
    ) -> float:
        confidence = torch.sigmoid(logit).flatten()
        pred_labels = (confidence > threshold).float().flatten()
        return (pred_labels == labels).sum().item() / len(labels)

    def get_auroc(self, logit: torch.Tensor, labels: torch.Tensor) -> float:
        confidence = torch.sigmoid(logit).flatten()
        return roc_auc_score(labels.flatten(), confidence)


class BinaryClassifierTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.logger = (
            logging.Logger("BinaryClassifierTrainer") if logger is None else logger
        )

    def make_bar_sentence(
        self,
        phase: str,
        epoch: int,
        total_step: int,
        step: int,
        eta: str,
        total_loss: float,
        metrics: Metrics,
    ) -> str:
        """ProgressBar의 stdout의 string을 생성하여 반환

        Args:
            phase (str): Epoch의 phase
            epoch (int): epoch
            total_step (int): total steps for one epoch
            step (int): Step (in a epoch)
            eta (str): Estimated Time of Arrival
            loss (float): loss
            metrics (Metrics): Metrics

        Returns:
            str: progressbar senetence

        """
        metric_sentence = "|".join(
            [f"{k}: {v:.4f}" for k, v in metrics.to_dict().items()]
        )

        return (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | total_loss: {total_loss:.4f} | "
            f"{metric_sentence}"
        )

    def run_epoch(
        self,
        phase: str,
        epoch: int,
        dataloader: torch.utils.data.DataLoader,
        threshold: float = 0.5,
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
        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        loss_meter = AverageMeter("loss")
        metrics_meter = Metrics()
        for step, batch in enumerate(dataloader):
            xs, ys = batch

            if phase == "train":
                self.model.train()
                logits = self.model(xs)
            else:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(xs)

            logits = logits.view(ys.shape)
            loss = self.loss(logits, ys.float())

            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # metric
            loss_meter.update(loss.item(), len(ys))

            confidences = torch.sigmoid(logits).flatten()
            flatten_ys = ys.flatten()
            metrics_meter.update(
                calculate_metrics(
                    confidences.detach().cpu().numpy(),
                    flatten_ys.detach().cpu().numpy(),
                    threshold=threshold,
                )
            )

            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=loss_meter.avg,
                metrics=metrics_meter,
            )
            bar.next()

        bar.finish()

        return (loss_meter, metrics_meter)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        n_patience: int,
    ):

        best_loss = math.inf
        for epoch in range(epochs):
            train_loss, train_metrics = self.run_epoch(
                phase="train", epoch=epoch, dataloader=train_dataloader
            )
            mlflow.log_metric("train_loss", train_loss.avg, step=epoch)
            mlflow.log_metrics(train_metrics.to_dict(prefix="train_"), step=epoch)

            val_loss, val_metrics = self.run_epoch(
                phase="validation", epoch=epoch, dataloader=val_dataloader
            )
            mlflow.log_metric("val_loss", val_loss.avg, step=epoch)
            mlflow.log_metrics(val_metrics.to_dict(prefix="val_"), step=epoch)

            if val_loss.avg < best_loss:
                best_loss = val_loss.avg
                patience = 0
            else:
                patience += 1
                if patience >= n_patience:
                    self.logger.info("Early stopping after epoch {}".format(epoch))
                    break

        return train_loss, train_metrics, val_loss, val_metrics
