"""Unit tests for Phase 2 — model, augmentation, dataset, loss, trainer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from backend.ml.data.augmentation import KeypointAugmentation
from backend.ml.data.dataset import SignSequenceDataset
from backend.ml.models.sign_model import SignBridgeModel
from backend.ml.training.losses import LabelSmoothingCrossEntropy
from backend.ml.training.trainer import Trainer

BATCH = 4
SEQ_LEN = 30
FEATURES = 130
NUM_CLASSES = 263


@pytest.fixture(scope="module")
def model() -> SignBridgeModel:
    return SignBridgeModel(num_classes=NUM_CLASSES, dropout=0.4)


@pytest.fixture
def random_input() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, FEATURES)


@pytest.fixture
def random_seq() -> np.ndarray:
    return np.random.rand(SEQ_LEN, FEATURES).astype(np.float32)


class TestModelForwardPass:
    def test_output_shape(self, model: SignBridgeModel, random_input: torch.Tensor) -> None:
        model.eval()
        with torch.no_grad():
            out = model(random_input)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_output_no_nan(self, model: SignBridgeModel, random_input: torch.Tensor) -> None:
        model.eval()
        with torch.no_grad():
            out = model(random_input)
        assert not torch.isnan(out).any(), "NaN values detected in model output."

    def test_output_no_inf(self, model: SignBridgeModel, random_input: torch.Tensor) -> None:
        model.eval()
        with torch.no_grad():
            out = model(random_input)
        assert not torch.isinf(out).any(), "Inf values detected in model output."

    def test_param_count_positive_int(self, model: SignBridgeModel) -> None:
        n = model.get_num_parameters()
        assert isinstance(n, int)
        assert n > 0

    def test_raises_on_invalid_num_classes(self) -> None:
        with pytest.raises(ValueError):
            SignBridgeModel(num_classes=0)

    def test_batch_size_invariant(self, model: SignBridgeModel) -> None:
        model.eval()
        for bs in [1, 8, 16]:
            x = torch.randn(bs, SEQ_LEN, FEATURES)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (bs, NUM_CLASSES)


class TestAugmentation:
    def test_time_warp_shape(self, random_seq: np.ndarray) -> None:
        out = KeypointAugmentation.time_warp(random_seq)
        assert out.shape == (SEQ_LEN, FEATURES)

    def test_mirror_shape(self, random_seq: np.ndarray) -> None:
        out = KeypointAugmentation.mirror_horizontal(random_seq)
        assert out.shape == (SEQ_LEN, FEATURES)

    def test_noise_shape(self, random_seq: np.ndarray) -> None:
        out = KeypointAugmentation.add_gaussian_noise(random_seq)
        assert out.shape == (SEQ_LEN, FEATURES)

    def test_random_apply_shape(self, random_seq: np.ndarray) -> None:
        out = KeypointAugmentation.random_apply(random_seq)
        assert out.shape == (SEQ_LEN, FEATURES)

    def test_mirror_swaps_hands(self, random_seq: np.ndarray) -> None:
        seq = random_seq.copy()
        seq[:, :63] = 1.0
        seq[:, 63:126] = 0.0
        mirrored = KeypointAugmentation.mirror_horizontal(seq)
        assert not np.allclose(mirrored[:, :63], seq[:, :63]), (
            "Mirror did not swap left hand region."
        )

    def test_noise_stays_in_range(self) -> None:
        seq = np.random.rand(SEQ_LEN, FEATURES).astype(np.float32)
        out = KeypointAugmentation.add_gaussian_noise(seq, std=0.5)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_time_warp_dtype(self, random_seq: np.ndarray) -> None:
        out = KeypointAugmentation.time_warp(random_seq)
        assert out.dtype == np.float32

    def test_mirror_dtype(self, random_seq: np.ndarray) -> None:
        out = KeypointAugmentation.mirror_horizontal(random_seq)
        assert out.dtype == np.float32


class TestDataset:
    @pytest.fixture
    def small_dataset(self) -> SignSequenceDataset:
        seqs = np.random.rand(100, SEQ_LEN, FEATURES).astype(np.float32)
        labels = np.array([i % 5 for i in range(100)], dtype=np.int64)
        return SignSequenceDataset(seqs, labels, augment=False)

    def test_len(self, small_dataset: SignSequenceDataset) -> None:
        assert len(small_dataset) == 100

    def test_getitem_shape(self, small_dataset: SignSequenceDataset) -> None:
        seq, label = small_dataset[0]
        assert seq.shape == (SEQ_LEN, FEATURES)
        assert label.shape == ()

    def test_getitem_dtype(self, small_dataset: SignSequenceDataset) -> None:
        seq, label = small_dataset[0]
        assert seq.dtype == torch.float32
        assert label.dtype == torch.int64

    def test_split_sizes(self) -> None:
        n = 200
        seqs = np.random.rand(n, SEQ_LEN, FEATURES).astype(np.float32)
        labels = np.array([i % 10 for i in range(n)])
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        enc = le.fit_transform(labels)
        x_tr, x_temp, y_tr, y_temp = train_test_split(seqs, enc, test_size=0.30, stratify=enc, random_state=42)
        x_v, x_te, y_v, y_te = train_test_split(x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
        assert len(x_tr) == pytest.approx(n * 0.70, abs=5)
        assert len(x_v) == pytest.approx(n * 0.15, abs=5)
        assert len(x_te) == pytest.approx(n * 0.15, abs=5)

    def test_no_label_leak(self) -> None:
        n = 300
        seqs = np.random.rand(n, SEQ_LEN, FEATURES).astype(np.float32)
        labels = np.array([i % 15 for i in range(n)])
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        enc = le.fit_transform(labels)
        x_tr, x_temp, y_tr, y_temp = train_test_split(seqs, enc, test_size=0.30, stratify=enc, random_state=42)
        x_v, x_te, y_v, y_te = train_test_split(x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
        assert set(np.unique(y_tr)) == set(np.unique(y_v)) == set(np.unique(y_te))


class TestLoss:
    def test_loss_is_scalar(self) -> None:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = criterion(logits, targets)
        assert loss.shape == ()

    def test_loss_positive(self) -> None:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = criterion(logits, targets)
        assert loss.item() > 0

    def test_loss_no_nan(self) -> None:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(8, 100)
        targets = torch.randint(0, 100, (8,))
        loss = criterion(logits, targets)
        assert not torch.isnan(loss)

    def test_loss_smoothing_zero_equals_ce(self) -> None:
        import torch.nn.functional as F
        criterion = LabelSmoothingCrossEntropy(smoothing=0.0)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        smoothed = criterion(logits, targets).item()
        ce = F.cross_entropy(logits, targets).item()
        assert abs(smoothed - ce) < 1e-4


class TestTrainerOneEpoch:
    def test_one_epoch_runs(self) -> None:
        n_samples = 40
        n_classes = 5
        seqs = np.random.rand(n_samples, SEQ_LEN, FEATURES).astype(np.float32)
        labels = np.array([i % n_classes for i in range(n_samples)], dtype=np.int64)

        train_ds = SignSequenceDataset(seqs[:28], labels[:28], augment=False)
        val_ds = SignSequenceDataset(seqs[28:], labels[28:], augment=False)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8)

        small_model = SignBridgeModel(num_classes=n_classes, dropout=0.1)

        config = {
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "epochs": 1,
                "early_stopping_patience": 5,
                "gradient_clip": 1.0,
                "label_smoothing": 0.1,
            },
            "scheduler": {"T_max": 1, "eta_min": 1e-5},
            "checkpoint": {
                "save_dir": "backend/models/",
                "best_model_name": "_test_best.pt",
                "save_every_n_epochs": 10,
            },
            "logging": {"run_dir": "runs/", "log_every_n_steps": 5},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)

        trainer = Trainer(
            model=small_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            language="ISL",
            config_path=config_path,
        )
        history = trainer.train()

        assert isinstance(history["train_loss"][0], float)
        assert 0.0 <= history["train_acc"][0] <= 1.0
        assert isinstance(history["val_loss"][0], float)
        assert 0.0 <= history["val_acc"][0] <= 1.0
