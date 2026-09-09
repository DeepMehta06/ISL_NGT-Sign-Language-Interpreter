"""Saves evaluation results as JSON, confusion matrix PNG, and chart PNGs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backend.core.logger import get_logger

logger = get_logger(__name__)

_REPORTS_DIR = Path(__file__).resolve().parents[0] / "reports"


class ReportGenerator:
    """Saves evaluation metrics to a timestamped report directory.

    Args:
        language: "ISL" or "NGT" — used for directory naming.
        label_names: List of class name strings.
    """

    def __init__(self, language: str, label_names: list[str]) -> None:
        self.language = language
        self.label_names = label_names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = _REPORTS_DIR / f"{language.lower()}_{timestamp}"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Report directory: {self.report_dir}")

    def save_all(self, metrics: dict, training_log_path: Path | None = None) -> None:
        """Save all evaluation artefacts.

        Args:
            metrics: Dict returned by Evaluator.evaluate().
            training_log_path: Optional path to the training CSV log for curves plot.
        """
        self.save_json(metrics)
        self.save_confusion_matrix(metrics["confusion_matrix"])
        self.save_per_class_accuracy(metrics["per_class_accuracy"])
        if training_log_path and training_log_path.exists():
            self.save_training_curves(training_log_path)

    def save_json(self, metrics: dict) -> None:
        """Save report.json with all scalar metrics.

        Args:
            metrics: Full metrics dict from Evaluator.evaluate().
        """
        report = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        out_path = self.report_dir / "report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved report.json: {out_path}")

    def save_confusion_matrix(self, cm: np.ndarray) -> None:
        """Save confusion matrix as a seaborn heatmap PNG.

        Truncates to the worst 50 classes by total error count for readability
        when num_classes > 50.

        Args:
            cm: Confusion matrix array of shape (num_classes, num_classes).
        """
        n = cm.shape[0]
        labels = self.label_names[:n]

        if n > 50:
            errors = cm.sum(axis=1) - np.diag(cm)
            worst_idx = np.argsort(errors)[-50:][::-1]
            cm = cm[np.ix_(worst_idx, worst_idx)]
            labels = [labels[i] for i in worst_idx]
            n = 50

        fig_size = max(12, n // 3)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            cm, annot=(n <= 20), fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(f"Confusion Matrix — {self.language}", fontsize=14)
        plt.tight_layout()

        out_path = self.report_dir / "confusion_matrix.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved confusion_matrix.png: {out_path}")

    def save_per_class_accuracy(self, per_class_acc: dict[str, float]) -> None:
        """Save bar chart of worst-20 classes by accuracy.

        Args:
            per_class_acc: Dict mapping class name to accuracy float.
        """
        sorted_items = sorted(per_class_acc.items(), key=lambda x: x[1])
        worst_20 = sorted_items[:20]
        names = [item[0] for item in worst_20]
        accs = [item[1] for item in worst_20]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, accs, color="#3ECFB2")
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_title(f"Worst 20 Classes by Accuracy — {self.language}", fontsize=14)
        ax.set_xlim(0, 1.0)
        for bar, acc in zip(bars, accs):
            ax.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.2f}", va="center", fontsize=9)
        plt.tight_layout()

        out_path = self.report_dir / "per_class_accuracy.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved per_class_accuracy.png: {out_path}")

    def save_training_curves(self, log_path: Path) -> None:
        """Save loss and accuracy curves from training CSV log.

        Args:
            log_path: Path to the training_log.csv file from Trainer.
        """
        df = pd.read_csv(log_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#3ECFB2")
        ax1.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#A78BFA")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(df["epoch"], df["train_acc"], label="Train Acc", color="#3ECFB2")
        ax2.plot(df["epoch"], df["val_acc"], label="Val Acc", color="#A78BFA")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.suptitle(f"Training Curves — {self.language}", fontsize=14)
        plt.tight_layout()

        out_path = self.report_dir / "training_curves.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved training_curves.png: {out_path}")
