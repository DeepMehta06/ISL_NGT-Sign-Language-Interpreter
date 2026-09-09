"""Training entry point for SignBridge.

Usage:
    python scripts/train.py --language ISL --config backend/ml/training/configs/isl_config.yaml

Flow:
    1. Load YAML config
    2. Build model via model_factory
    3. Build datasets via dataset.build_datasets (real .npy files required)
    4. Run Trainer.train()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ml.data.dataset import build_dataloaders, build_datasets
from backend.ml.models.model_factory import build_model
from backend.ml.training.trainer import Trainer

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SignBridge model.")
    parser.add_argument("--language", choices=["ISL", "NGT"], required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file relative to signbridge/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (
        settings.project_root / args.config
    )

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config: {config_path}")

    data_cfg = config["data"]
    sequences_path = settings.project_root / data_cfg["sequences_path"]
    labels_path = settings.project_root / data_cfg["labels_path"]

    train_ds, val_ds, test_ds, label_encoder = build_datasets(
        sequences_path=sequences_path,
        labels_path=labels_path,
        language=args.language,
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        augment_train=data_cfg["augment_train"],
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=config["training"]["batch_size"],
    )

    model = build_model(config, language=args.language)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        language=args.language,
        config_path=config_path,
    )
    trainer.train()


if __name__ == "__main__":
    main()
