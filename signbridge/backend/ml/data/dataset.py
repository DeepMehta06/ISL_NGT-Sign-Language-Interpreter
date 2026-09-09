"""PyTorch Dataset for sign language keypoint sequences.

Loads preprocessed numpy arrays, applies stratified 70/15/15 splits,
fits zero-mean unit-variance normalisation on train set only,
and applies augmentation during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ml.data.augmentation import KeypointAugmentation

logger = get_logger(__name__)


class SignSequenceDataset(Dataset):
    """PyTorch Dataset for sign language keypoint sequences.

    Loads preprocessed numpy arrays from data/processed/.
    Applies optional augmentation during training.

    Args:
        sequences: Float32 array of shape (N, window_size, num_keypoints).
        labels: Integer array of shape (N,) — encoded class indices.
        augment: Whether to apply KeypointAugmentation on __getitem__.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.sequences = sequences.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()
        if self.augment:
            seq = KeypointAugmentation.random_apply(seq)
        return torch.from_numpy(seq), torch.tensor(self.labels[idx], dtype=torch.long)


def build_datasets(
    sequences_path: Path,
    labels_path: Path,
    language: str,
    train_split: float = 0.70,
    val_split: float = 0.15,
    augment_train: bool = True,
    label_encoder: Optional[LabelEncoder] = None,
) -> tuple[SignSequenceDataset, SignSequenceDataset, SignSequenceDataset, LabelEncoder]:
    """Load .npy arrays, split, normalise, and return Dataset objects.

    Normalisation stats (mean, std) are computed on the training set only
    and applied identically to val and test sets. Stats are saved to
    data/processed/norm_stats_{language}.npy for inference use.

    Args:
        sequences_path: Path to sequences .npy file, shape (N, 30, 130).
        labels_path: Path to labels .npy file, shape (N,) strings.
        language: "ISL" or "NGT" — used for naming norm_stats file.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        augment_train: Whether to augment training samples.
        label_encoder: Optional pre-fitted LabelEncoder. Fitted on labels if None.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, label_encoder).

    Raises:
        FileNotFoundError: If sequences_path or labels_path does not exist.
    """
    if not sequences_path.exists():
        raise FileNotFoundError(
            f"Sequences file not found: {sequences_path}\n"
            "Run: python scripts/prepare_data.py --dataset include"
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            "Run: python scripts/prepare_data.py --dataset include"
        )

    sequences = np.load(sequences_path)
    raw_labels = np.load(labels_path, allow_pickle=True)

    logger.info(f"Loaded sequences: {sequences.shape}, labels: {raw_labels.shape}")

    if label_encoder is None:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(raw_labels)
    else:
        encoded_labels = label_encoder.transform(raw_labels)

    test_size = 1.0 - train_split
    val_ratio_of_remainder = val_split / (val_split + (1.0 - train_split - val_split))

    x_train, x_temp, y_train, y_temp = train_test_split(
        sequences, encoded_labels,
        test_size=test_size,
        stratify=encoded_labels,
        random_state=42,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    logger.info(
        f"Split sizes — train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}"
    )

    # Normalise: fit on train, apply to all splits
    mean = x_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, 130)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)  # avoid div-by-zero for zero-filled features

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    norm_path = settings.processed_data_dir / f"norm_stats_{language.lower()}.npy"
    np.save(norm_path, np.array({"mean": mean, "std": std}))
    logger.info(f"Normalisation stats saved: {norm_path}")

    return (
        SignSequenceDataset(x_train, y_train, augment=augment_train),
        SignSequenceDataset(x_val, y_val, augment=False),
        SignSequenceDataset(x_test, y_test, augment=False),
        label_encoder,
    )


def build_dataloaders(
    train_ds: SignSequenceDataset,
    val_ds: SignSequenceDataset,
    test_ds: SignSequenceDataset,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap datasets in DataLoaders.

    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        test_ds: Test dataset.
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
