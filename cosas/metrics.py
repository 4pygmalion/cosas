from __future__ import annotations

import math
from typing import List, Tuple, Dict
from itertools import cycle
from dataclasses import dataclass, asdict, field


import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
)


class AverageMeter:
    """Computes and stores the average and current value"""

    name: str

    def __init__(self, name: str):
        self.name = name
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: float = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name}, avg={self.avg:.4f}, count={self.count})"


@dataclass
class Metrics:
    f1: AverageMeter = None
    acc: AverageMeter = None
    sen: AverageMeter = None
    spec: AverageMeter = None
    auroc: AverageMeter = None
    prauc: AverageMeter = None

    def __post_init__(self):
        self.f1 = AverageMeter("f1")
        self.acc = AverageMeter("acc")
        self.sen = AverageMeter("sen")
        self.spec = AverageMeter("spec")
        self.auroc = AverageMeter("auroc")
        self.prauc = AverageMeter("prauc")

    def update(self, metrics: Dict[str, float], n: int = 1):
        for key, value in metrics.items():
            if hasattr(self, key):
                meter = getattr(self, key)
                meter.update(value, n)

            else:
                meter = AverageMeter(key)
                meter.update(value, n)
                setattr(self, key, meter)

    def to_dict(self, prefix=str()):
        return {
            prefix + attr: round(meter.avg, 5)
            for attr, meter in self.__dict__.items()
            if isinstance(meter, AverageMeter)
        }


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over Union 계산"""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = intersection / union if union != 0 else 0.0
    return iou


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """DICE score 계산"""
    intersection = np.logical_and(pred, target).sum()
    dice = (
        (2 * intersection) / (pred.sum() + target.sum())
        if (pred.sum() + target.sum()) != 0
        else 0.0
    )
    return dice


def calculate_metrics(
    confidences: np.ndarray, targets: np.ndarray, threshold=0.5
) -> Dict[str, float]:
    pred_confidence = confidences.flatten()
    pred_label = pred_confidence >= threshold
    target_flat = targets.flatten()

    f1 = f1_score(target_flat, pred_label)
    acc = accuracy_score(target_flat, pred_label)
    sen = recall_score(target_flat, pred_label)
    spec = specificity_score(target_flat, pred_label)
    auroc = (
        roc_auc_score(target_flat, pred_confidence)
        if len(np.unique(target_flat)) > 1
        else 0.0
    )
    pr, rc, _ = precision_recall_curve(target_flat, pred_confidence)
    prauc = auc(rc, pr)
    iou = compute_iou(pred_label, target_flat)
    dice = compute_dice(pred_label, target_flat)

    return {
        "f1": f1,
        "acc": acc,
        "sen": sen,
        "spec": spec,
        "auroc": auroc,
        "prauc": prauc,
        "iou": iou,
        "dice": dice,
    }


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    return specificity
