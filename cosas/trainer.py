import math
import logging
from copy import deepcopy
from typing import Tuple, Literal
from abc import ABC, abstractmethod

import mlflow
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from progress.bar import Bar
from sklearn.metrics import roc_auc_score

from .tracking import log_patch_and_save_by_batch
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
        device: str = "cuda",
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
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
        metric_sentence = " | ".join(
            [f"{k}: {v:.4f}" for k, v in metrics.to_dict().items()]
        )

        return (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | EPOCH status ==> "
            f"total_loss: {total_loss:.4f} | "
            f"{metric_sentence}"
        )

    def run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        phase: Literal["train", "val", "test"],
        threshold: float = 0.5,
        save_plot: bool = False,
        update_step: int = 1,
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
            update_step: 훈련 중 각 업데이트 사이의 단계 수. 1의 값은 매 단계마다 업데이트를 의미합니다 (기본값).
                더 높은 값은 그래디언트 누적에 사용될 수 있습니다.

        Returns:
            Tuple: loss, accuracy, top_k_recall
        """

        # init
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        epoch_metrics = Metrics()
        loss_meter = AverageMeter("loss")

        for step, batch in enumerate(dataloader, start=1):
            xs, ys = batch
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            if phase == "train":
                logits = self.model(xs)
                logits = logits.view(ys.shape)
                loss = self.loss(logits, ys.float())
                loss.backward()
                if step % update_step == 0 or step == len(dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            else:
                with torch.no_grad():
                    logits = self.model(xs)
                    logits = logits.view(ys.shape)
                    loss = self.loss(logits, ys.float())

            # metric
            loss_meter.update(loss.item(), len(ys))

            logits = torch.clip(logits, -1e10, 1e10)
            confidences = torch.sigmoid(logits)
            flat_confidence = confidences.flatten().detach().cpu().numpy()
            ground_truths: torch.Tensor = ys.flatten().detach().cpu().numpy()

            epoch_metrics.update(
                calculate_metrics(
                    flat_confidence,
                    ground_truths,
                    threshold=threshold,
                )
            )

            if save_plot:
                log_patch_and_save_by_batch(xs, ys, confidences, phase=phase)

            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=loss_meter.avg,
                metrics=epoch_metrics,
            )
            bar.next()

        bar.finish()

        return (loss_meter, epoch_metrics)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        n_patience: int,
        update_step: int = 1,
    ):

        best_state_dict = deepcopy(self.model.state_dict())
        best_loss = math.inf
        patience = 0
        for epoch in range(epochs):
            train_loss, train_metrics = self.run_epoch(
                dataloader=train_dataloader,
                epoch=epoch,
                phase="train",
                update_step=update_step,
            )
            mlflow.log_metric("train_loss", train_loss.avg, step=epoch)
            mlflow.log_metrics(train_metrics.to_dict(prefix="train_"), step=epoch)

            val_loss, val_metrics = self.run_epoch(
                dataloader=val_dataloader, epoch=epoch, phase="val"
            )
            mlflow.log_metric("val_loss", val_loss.avg, step=epoch)
            mlflow.log_metrics(val_metrics.to_dict(prefix="val_"), step=epoch)

            val_loss = val_loss.avg
            if val_loss <= best_loss:
                best_loss = val_loss
                patience = 0
                best_state_dict = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= n_patience:
                    self.logger.info("Early stopping after epoch {}".format(epoch))
                    break

        self.model.load_state_dict(best_state_dict)
        self.run_epoch(
            dataloader=train_dataloader, epoch=epoch, phase="train_save", save_plot=True
        )
        self.run_epoch(
            dataloader=val_dataloader, epoch=epoch, phase="val_save", save_plot=True
        )

        return train_loss, train_metrics, val_loss, val_metrics


class SSLTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        device: str = "cuda",
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
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
        metrics: dict,
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
        metric_sentence = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        return (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | EPOCH status ==> "
            f"total_loss: {total_loss:.4f} | "
            f"{metric_sentence}"
        )

    def run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        phase: Literal["train", "val", "test"],
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

        # init
        if phase == "train":
            self.model.train()
        else:
            self.model.eval

        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        loss_meter = AverageMeter("loss")
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=0.001,
            max_lr=0.1,
            step_size_up=int(len(dataloader)) * 4,
            step_size_down=int(len(dataloader)) * 4,
        )

        for step, batch in enumerate(dataloader):
            xs, ys = batch
            batch_size, n_views, c, w, h = xs.shape
            xs = xs.to(self.device).view(-1, 3, w, h)
            ys = ys.to(self.device)

            if phase == "train":
                features = self.model(xs)
                *_, ndim = features.shape
                features = features.view(batch_size, n_views, ndim)
                loss = self.loss(features, ys.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

            else:
                with torch.no_grad():
                    features = self.model(xs)
                    *_, ndim = features.shape
                    features = features.view(batch_size, n_views, ndim)
                    loss = self.loss(features, ys.float())

            # metric
            loss_meter.update(loss.item(), len(ys))
            metrics = {f"{phase}_loss": loss_meter.avg}
            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=loss_meter.avg,
                metrics=metrics,
            )
            bar.next()

        bar.finish()

        return (loss_meter, metrics)

    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int,
    ):
        for epoch in range(epochs):
            train_loss, _ = self.run_epoch(
                dataloader=train_dataloader, epoch=epoch, phase="train"
            )
            mlflow.log_metric("train_loss", train_loss.avg, step=epoch)

        mlflow.pytorch.log_model(self.model, "encoder")

        return train_loss


class AETrainer(BinaryClassifierTrainer):
    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        device: str = "cuda",
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.logger = logging.Logger("AutoEncoderTrainer") if logger is None else logger

    def run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        phase: Literal["train", "val", "test"],
        threshold: float = 0.5,
        save_plot: bool = False,
        **kwargs,
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

        # init
        if phase == "train":
            self.model.train()
        else:
            self.model.eval

        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        epoch_metrics = Metrics()
        loss_meter = AverageMeter("loss")
        i = 0
        for step, batch in enumerate(dataloader):
            xs, ys = batch
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            if phase == "train":
                outputs = self.model(xs)
                recon_x = outputs["recon"]
                logits = outputs["mask"]
                vector = outputs["vector"]
                density = outputs["density"]

                logits = logits.view(ys.shape)
                loss = self.loss(recon_x, xs, logits, ys.float(), vector, density)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            else:
                with torch.no_grad():
                    outputs = self.model(xs)
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
                log_patch_and_save_by_batch(xs, ys, images_confidences, phase=phase)

            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=loss_meter.avg,
                metrics=epoch_metrics,
            )
            bar.next()

        bar.finish()

        return (loss_meter, epoch_metrics)


class MultiTaskBinaryClassifierTrainer(BinaryClassifierTrainer):
    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        device: str = "cuda",
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.logger = logging.Logger("AutoEncoderTrainer") if logger is None else logger

    def run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        phase: Literal["train", "val", "test"],
        threshold: float = 0.5,
        save_plot: bool = False,
        **kwargs,
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

        # init
        if phase == "train":
            self.model.train()
        else:
            self.model.eval

        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        epoch_metrics = Metrics()
        loss_meter = AverageMeter("loss")

        for step, (xs, ys, ys_density) in enumerate(dataloader):
            xs = xs.to(self.device)
            ys = ys.to(self.device)
            ys_density = ys_density.to(self.device)

            if phase == "train":
                outputs = self.model(xs)
                logits = outputs["mask"]
                density = outputs["density"]

                logits = logits.view(ys.shape)
                loss = self.loss(logits, ys.float(), density, ys_density)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            else:
                with torch.no_grad():
                    outputs = self.model(xs)
                    logits = outputs["mask"]
                    density = outputs["density"]
                    logits = logits.view(ys.shape)
                    loss = self.loss(logits, ys.float(), density, ys_density)

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
                log_patch_and_save_by_batch(xs, ys, images_confidences, phase=phase)

            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=loss_meter.avg,
                metrics=epoch_metrics,
            )
            bar.next()

        bar.finish()

        return (loss_meter, epoch_metrics)
