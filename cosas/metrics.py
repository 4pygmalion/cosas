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
    cosas_score: AverageMeter = None

    def __post_init__(self):
        self.f1 = AverageMeter("f1")
        self.acc = AverageMeter("acc")
        self.sen = AverageMeter("sen")
        self.spec = AverageMeter("spec")
        self.auroc = AverageMeter("auroc")
        self.prauc = AverageMeter("prauc")
        self.cosas_score = AverageMeter("cosas_score")

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
    if pred.sum() + target.sum() == 0:
        return 1.0

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = intersection / union if union != 0 else 0.0
    return iou


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """DICE score 계산"""
    if pred.sum() + target.sum() == 0:
        return 1.0

    intersection = np.logical_and(pred, target).sum()
    dice = (
        (2 * intersection) / (pred.sum() + target.sum())
        if (pred.sum() + target.sum()) != 0
        else 0.0
    )
    return dice


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    if np.unique(y_true).tolist() == [1]:
        return 0

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        if sum(y_true) + sum(y_pred) == 0:
            tn = int(confusion_matrix(y_true, y_pred).ravel())
            fp = 0
            fn = 0
            tp = 0
    finally:
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        return specificity


def calculate_metrics(
    confidences: np.ndarray,
    targets: np.ndarray,
    threshold=0.5,
    postprocess: callable = None,
) -> Dict[str, float]:

    pred_label = (confidences >= threshold).astype(np.uint8)
    if postprocess:
        pred_label = postprocess(pred_label).astype(np.uint8)

    pred_label = pred_label.ravel()
    breakpoint()
    f1 = f1_score(targets, pred_label)
    acc = accuracy_score(targets, pred_label)

    # Calculate specificity
    spec = specificity_score(targets, pred_label) if targets.sum() > 0 else 1.0

    # Calculate sensitivity
    sen = recall_score(targets, pred_label) if targets.sum() > 0 else 0.0

    # Calculate AUROC
    auroc = roc_auc_score(targets, confidences) if len(np.unique(targets)) > 1 else 0.0

    # Calculate Precision-Recall AUC
    prauc = 0.0
    if targets.sum() > 0:
        pr, rc, _ = precision_recall_curve(targets, confidences)
        prauc = auc(rc, pr)

    # Calculate IoU and Dice
    iou = compute_iou(pred_label, targets)
    dice = compute_dice(pred_label, targets)

    return {
        "f1": f1,
        "acc": acc,
        "sen": sen,
        "spec": spec,
        "auroc": auroc,
        "prauc": prauc,
        "iou": iou,
        "dice": dice,
        "cosas_score": (iou + dice) / 2,
    }


def summarize_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate the average value for each key in a list of dictionaries.

    This function takes a list of dictionaries where each dictionary contains
    metrics with float values. It calculates the average value for each key
    across all dictionaries in the list.

    Parameters:
    metrics (List[Dict[str, float]]): A list of dictionaries, where each dictionary
                                      contains metric names as keys and float values
                                      as their corresponding values.

    Returns:
    Dict[str, float]: A dictionary containing the average value for each key
                      across the input list of dictionaries.

    Example:
    >>> metrics = [
    ...     {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.7},
    ...     {'accuracy': 0.82, 'precision': 0.78, 'recall': 0.72},
    ...     {'accuracy': 0.81, 'precision': 0.76, 'recall': 0.74},
    ... ]
    >>> summarize_metrics(metrics)
    {'accuracy': 0.81, 'precision': 0.7633333333333333, 'recall': 0.72}
    """
    keys = metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        values = [metric[key] for metric in metrics]
        avg_metrics[key] = sum(values) / len(values)
    return avg_metrics
