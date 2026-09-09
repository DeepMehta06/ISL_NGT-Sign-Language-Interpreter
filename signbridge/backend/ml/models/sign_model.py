"""CNN + Bidirectional LSTM model for sign language recognition.

Architecture:
    Input [B, 30, 130]
    -> Spatial Encoder: 3x Conv1d + BN + ReLU + Dropout -> [B, 30, 256]
    -> Temporal Encoder: BiLSTM(256, hidden=256, layers=2) -> [B, 512]
    -> Classifier: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> [B, C]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SignBridgeModel(nn.Module):
    """CNN + Bidirectional LSTM sign language recognition model.

    Args:
        num_classes: Number of output sign classes.
        dropout: Dropout probability applied in CNN and classifier head.
        lstm_dropout: Dropout probability applied between LSTM layers.
        hidden_size: LSTM hidden state size per direction.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.4,
        lstm_dropout: float = 0.3,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Spatial encoder — Conv1d operates on [B, features, time]
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(in_channels=130, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Temporal encoder — bidirectional LSTM
        self.temporal_encoder = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )

        # Classifier head — input is 512 (256 forward + 256 backward)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape (batch_size, window_size, num_keypoints)
               i.e. (B, 30, 130).

        Returns:
            Raw logits tensor of shape (batch_size, num_classes). No softmax applied.
        """
        # x: [B, T, F] -> permute to [B, F, T] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.spatial_encoder(x)

        # [B, 256, T] -> [B, T, 256] for LSTM
        x = x.permute(0, 2, 1)

        # LSTM: output all hidden states, use only final hidden state
        _, (hidden, _) = self.temporal_encoder(x)
        # hidden: [num_layers * 2, B, hidden_size]
        # Take the last layer's forward and backward hidden states
        forward_hidden = hidden[-2]   # [B, hidden_size]
        backward_hidden = hidden[-1]  # [B, hidden_size]
        x = torch.cat([forward_hidden, backward_hidden], dim=1)  # [B, 512]

        return self.classifier(x)

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Integer count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
