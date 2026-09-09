"""Full training loop for SignBridge sign language model."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ml.training.losses import LabelSmoothingCrossEntropy

logger = get_logger(__name__)

_LOG_COLUMNS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]


class Trainer:
    """Training loop with validation, early stopping, and checkpoint saving.

    Args:
        model: Initialised SignBridgeModel (on CPU or GPU).
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Parsed YAML config dict.
        language: "ISL" or "NGT" — used for run directory naming.
        config_path: Path to the original YAML file — copied into run dir.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        language: str,
        config_path: Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.language = language

        train_cfg = config["training"]
        sched_cfg = config["scheduler"]
        ckpt_cfg = config["checkpoint"]
        log_cfg = config["logging"]

        self.epochs: int = train_cfg["epochs"]
        self.patience: int = train_cfg["early_stopping_patience"]
        self.grad_clip: float = train_cfg["gradient_clip"]
        self.log_every: int = log_cfg["log_every_n_steps"]
        self.save_every: int = ckpt_cfg["save_every_n_epochs"]

        self.device = torch.device(settings.device)
        self.model = self.model.to(self.device)

        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=train_cfg["label_smoothing"]
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=sched_cfg["T_max"],
            eta_min=sched_cfg["eta_min"],
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_root = settings.project_root / log_cfg["run_dir"]
        self.run_dir = runs_root / f"{language.lower()}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(config_path, self.run_dir / config_path.name)
        logger.info(f"Run directory: {self.run_dir}")

        self.log_path = self.run_dir / "training_log.csv"
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_LOG_COLUMNS).writeheader()

        save_dir = settings.project_root / ckpt_cfg["save_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = save_dir / ckpt_cfg["best_model_name"]
        self.periodic_ckpt_dir = self.run_dir / "checkpoints"
        self.periodic_ckpt_dir.mkdir(exist_ok=True)

        self._best_val_acc: float = 0.0
        self._no_improve_count: int = 0
        self._stopped_early: bool = False

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            Dict with keys "train_loss", "train_acc", "val_loss", "val_acc", "lr"
            each mapping to a list of per-epoch values.
        """
        history: dict[str, list[float]] = {k: [] for k in _LOG_COLUMNS if k != "epoch"}

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate(epoch)
            current_lr = self.scheduler.get_last_lr()[0]

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.6f}"
            )

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": round(current_lr, 8),
            }
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=_LOG_COLUMNS).writerow(row)

            for k in history:
                history[k].append(row[k])

            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                self._no_improve_count = 0
                self._save_checkpoint(self.best_ckpt_path, epoch, val_acc, is_best=True)
            else:
                self._no_improve_count += 1

            if epoch % self.save_every == 0:
                periodic_path = self.periodic_ckpt_dir / f"epoch_{epoch:04d}.pt"
                self._save_checkpoint(periodic_path, epoch, val_acc)

            if self._no_improve_count >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)."
                )
                self._stopped_early = True
                break

        logger.info(
            f"Training complete. Best val_acc={self._best_val_acc:.4f} | "
            f"{'Early stopped' if self._stopped_early else 'Full training'}."
        )
        return history

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch number (1-indexed).

        Returns:
            Tuple of (mean_loss, accuracy) for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for step, (sequences, labels) in enumerate(pbar, 1):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(sequences)
            loss = self.criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if step % self.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader), correct / total

    def _validate(self, epoch: int) -> tuple[float, float]:
        """Run one validation pass.

        Args:
            epoch: Current epoch number (for logging).

        Returns:
            Tuple of (mean_loss, accuracy) on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in tqdm(self.val_loader, desc=f"Epoch {epoch} [val]", leave=False):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_acc: float,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint with metadata.

        Args:
            path: Destination .pt file path.
            epoch: Current epoch number.
            val_acc: Validation accuracy at time of saving.
            is_best: Whether this is the best checkpoint so far.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "language": self.language,
                "config": self.config,
            },
            path,
        )
        tag = " [BEST]" if is_best else ""
        logger.info(f"Checkpoint saved{tag}: {path} | epoch={epoch} val_acc={val_acc:.4f}")
