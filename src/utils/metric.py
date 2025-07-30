import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from typing import List, Union, Dict, Optional, Any


class MetricEvaluator:
    """
    Evaluates classification metrics including accuracy, AUROC, F1-score,
    precision, recall, and optionally per-class metrics.
    """
    def __init__(self, num_classes: Optional[int] = None) -> None:
        """
        Args:
            num_classes (Optional[int]): Number of classes in the classification task.
        """
        self.num_classes = num_classes

    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes top-1 accuracy between model outputs and ground truth.

        Args:
            outputs (torch.Tensor): Raw model outputs (logits), shape (N, C).
            targets (torch.Tensor): Ground-truth labels, shape (N,).

        Returns:
            float: Accuracy value between 0 and 1.
        """
        preds = outputs.argmax(dim=1)
        correct = targets.eq(preds).sum().item()
        return correct / targets.size(0)

    def calc_metrics(
        self,
        y_true: List[int],
        y_score: Union[np.ndarray, List[List[float]]],
        y_pred: List[int],
        return_per_class: bool = False,
        apply_softmax: bool = True,
    ) -> Dict[str, Any]:
        """
        Computes classification metrics including AUROC, F1, recall, precision.
        Optionally includes per-class metrics and confusion matrix.

        Args:
            y_true (List[int]): Ground-truth class indices.
            y_score (Union[np.ndarray, List[List[float]]]): Raw model output scores or logits.
            y_pred (List[int]): Predicted class indices.
            return_per_class (bool): Whether to include per-class metrics.
            apply_softmax (bool): Whether to apply softmax to y_score before computing AUROC.

        Returns:
            Dict[str, Any]: Dictionary of computed metrics.
        """
        y_score_tensor = torch.FloatTensor(y_score)
        if apply_softmax:
            y_score_tensor = torch.nn.functional.softmax(y_score_tensor, dim=1)
        y_score_np = y_score_tensor.numpy()

        metrics = {
            "auroc": roc_auc_score(
                y_true, y_score_np, average="macro", multi_class="ovr"
            ),
            "f1": f1_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "precision": precision_score(y_true, y_pred, average="macro"),
        }

        if return_per_class:
            cm = confusion_matrix(y_true, y_pred)
            metrics["per_class"] = {
                "confusion_matrix": cm.tolist(),
                "f1": f1_score(y_true, y_pred, average=None).tolist(),
                "recall": recall_score(y_true, y_pred, average=None).tolist(),
                "precision": precision_score(y_true, y_pred, average=None).tolist(),
                "accuracy": (cm.diagonal() / cm.sum(axis=1)).tolist(),
            }

        return metrics


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count