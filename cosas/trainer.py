import math
import logging
from copy import deepcopy
from typing import Tuple, Literal
from abc import ABC, abstractmethod

import mlflow
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from progress.bar import Bar
from sklearn.metrics import roc_auc_score

from .tracking import plot_and_save, log_patch_and_save
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
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = "cuda",
        logger: logging.Logger = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=0.001,
            max_lr=0.1,
            step_size_up=int(len(dataloader)) * 4,
            step_size_down=int(len(dataloader)) * 4,
        )
        i = 0
        for step, batch in enumerate(dataloader):
            xs, ys = batch
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            if phase == "train":
                logits = self.model(xs)
                logits = logits.view(ys.shape)
                loss = self.loss(logits, ys.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

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
                for x, y, patch_confidence in zip(xs, ys, confidences):
                    mean = [0.485, 0.456, 0.406]
                    sd = [0.229, 0.224, 0.225]
                    original_x = ToPILImage()(
                        x.detach().cpu() * torch.tensor(sd)[:, None, None]
                        + torch.tensor(mean)[:, None, None]
                    )
                    log_patch_and_save(
                        image_name=f"step_{i}",
                        original_x=np.array(original_x),
                        original_y=y.detach().cpu().numpy(),
                        pred_masks=patch_confidence.detach().cpu().numpy() >= 0.5,
                        artifact_dir=f"{phase}_prediction",
                    )
                    i += 1

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
    ):

        best_state_dict = deepcopy(self.model.state_dict())
        best_loss = math.inf
        patience = 0
        for epoch in range(epochs):
            train_loss, train_metrics = self.run_epoch(
                dataloader=train_dataloader, epoch=epoch, phase="train"
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
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = "cuda",
        logger: logging.Logger = None,
    ):
        super(AETrainer, self).__init__(
            model, loss, optimizer, scheduler, device, logger
        )

    def run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        phase: Literal["train", "val", "test"],
        threshold: float = 0.5,
        save_plot: bool = False,
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
                if self.scheduler:
                    self.scheduler.step()

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
                        artifact_dir=f"{phase}_prediction",
                    )
                    i += 1

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


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
