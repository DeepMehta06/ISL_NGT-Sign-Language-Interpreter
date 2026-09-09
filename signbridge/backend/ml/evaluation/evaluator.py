"""Evaluation metrics for SignBridge model assessment."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Computes evaluation metrics for a trained SignBridgeModel on a test set.

    Args:
        model: Trained model in eval mode.
        test_loader: DataLoader for the held-out test set.
        label_names: List of class name strings (from LabelEncoder.classes_).
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        label_names: list[str],
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.label_names = label_names
        self.device = torch.device(settings.device)

    def evaluate(self) -> dict:
        """Run inference on the test set and compute all metrics.

        Returns:
            Dict with keys:
                top1_accuracy, top5_accuracy, macro_f1, weighted_f1,
                per_class_accuracy (dict: class_name -> accuracy),
                confusion_matrix (np.ndarray),
                classification_report (str).
        """
        all_preds, all_labels, all_probs = self._run_inference()

        num_classes = len(self.label_names)
        top1 = float((all_preds == all_labels).mean())

        top5_k = min(5, num_classes)
        top5 = top_k_accuracy_score(all_labels, all_probs, k=top5_k)

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

        per_class_acc = {}
        for cls_idx in range(num_classes):
            mask = all_labels == cls_idx
            if mask.sum() > 0:
                acc = float((all_preds[mask] == all_labels[mask]).mean())
            else:
                acc = 0.0
            per_class_acc[self.label_names[cls_idx]] = round(acc, 4)

        clf_report = classification_report(
            all_labels, all_preds,
            target_names=self.label_names,
            zero_division=0,
        )

        logger.info(
            f"Evaluation complete | top1={top1:.4f} top5={top5:.4f} "
            f"macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}"
        )

        return {
            "top1_accuracy": round(top1, 6),
            "top5_accuracy": round(top5, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm,
            "classification_report": clf_report,
            "num_classes": num_classes,
            "num_test_samples": len(all_labels),
        }

    def _run_inference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect predictions and labels across the full test loader.

        Returns:
            Tuple of (predictions, labels, probabilities) all as numpy arrays.
        """
        self.model.eval()
        self.model.to(self.device)

        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for sequences, labels in tqdm(self.test_loader, desc="Evaluating"):
                sequences = sequences.to(self.device)
                logits = self.model(sequences)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.numpy())
                all_probs.append(probs)

        return (
            np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs),
        )
