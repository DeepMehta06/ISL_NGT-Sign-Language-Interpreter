"""Evaluation entry point for SignBridge.

Usage:
    python scripts/evaluate.py --language ISL --checkpoint backend/models/best_isl.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ml.data.dataset import build_dataloaders, build_datasets
from backend.ml.evaluation.evaluator import Evaluator
from backend.ml.evaluation.report import ReportGenerator
from backend.ml.models.model_factory import build_model

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SignBridge checkpoint.")
    parser.add_argument("--language", choices=["ISL", "NGT"], required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=None,
        help="Optional path to training_log.csv for training curves plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else (
        settings.project_root / args.checkpoint
    )

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    data_cfg = config["data"]
    sequences_path = settings.project_root / data_cfg["sequences_path"]
    labels_path = settings.project_root / data_cfg["labels_path"]

    _, _, test_ds, label_encoder = build_datasets(
        sequences_path=sequences_path,
        labels_path=labels_path,
        language=args.language,
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        augment_train=False,
    )
    _, _, test_loader = build_dataloaders(
        test_ds, test_ds, test_ds,
        batch_size=config["training"]["batch_size"],
    )

    model = build_model(config, language=args.language)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        label_names=list(label_encoder.classes_),
    )
    metrics = evaluator.evaluate()

    reporter = ReportGenerator(language=args.language, label_names=list(label_encoder.classes_))
    reporter.save_all(metrics, training_log_path=args.training_log)

    logger.info(
        f"Results saved to: {reporter.report_dir}\n"
        f"  top1_accuracy = {metrics['top1_accuracy']:.4f}\n"
        f"  top5_accuracy = {metrics['top5_accuracy']:.4f}\n"
        f"  macro_f1      = {metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
