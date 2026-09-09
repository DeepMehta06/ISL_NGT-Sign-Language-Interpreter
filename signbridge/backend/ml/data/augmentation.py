"""Keypoint sequence augmentation for SignBridge training data."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from backend.core.config import settings


class KeypointAugmentation:
    """Augmentation transforms for (window_size, num_keypoints) sequences.

    All methods are static and return a new array of the same shape.
    All transforms preserve shape (30, 130) exactly.
    """

    @staticmethod
    def time_warp(seq: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Randomly stretch/compress the time axis using a cubic spline warp.

        Args:
            seq: Array of shape (window_size, num_keypoints).
            sigma: Controls warp intensity. Higher = more distortion.

        Returns:
            Warped array of the same shape, resampled back to window_size frames.
        """
        window_size = seq.shape[0]
        orig_steps = np.arange(window_size)

        # Generate random warp knots
        num_knots = 4
        knot_x = np.linspace(0, window_size - 1, num_knots + 2)
        knot_y = knot_x + np.random.normal(0, sigma * window_size, len(knot_x))
        knot_y[0] = 0
        knot_y[-1] = window_size - 1

        # Fit spline and generate warped time steps
        cs = CubicSpline(knot_x, knot_y)
        warped_steps = np.clip(cs(orig_steps), 0, window_size - 1)

        # Resample each feature dimension along warped time axis
        warped = np.zeros_like(seq)
        for feat in range(seq.shape[1]):
            feat_cs = CubicSpline(orig_steps, seq[:, feat])
            warped[:, feat] = feat_cs(warped_steps)

        return warped.astype(np.float32)

    @staticmethod
    def mirror_horizontal(seq: np.ndarray) -> np.ndarray:
        """Flip x-coordinates and swap left/right hand landmark blocks.

        Flips x values as x_new = 1.0 - x for all hand landmarks.
        Swaps left hand block (indices 0-62) with right hand block (63-125).
        Pose context (126-129) x-coords are also flipped.

        Args:
            seq: Array of shape (window_size, num_keypoints).

        Returns:
            Mirrored array of the same shape.
        """
        mirrored = seq.copy()

        left = mirrored[:, :63].copy()
        right = mirrored[:, 63:126].copy()

        # Flip x-coordinates (every 3rd value starting at 0 within each block)
        left[:, 0::3] = 1.0 - left[:, 0::3]
        right[:, 0::3] = 1.0 - right[:, 0::3]

        # Swap hands
        mirrored[:, :63] = right
        mirrored[:, 63:126] = left

        # Flip pose context x-coords (all 4 values are x-only)
        mirrored[:, 126:] = 1.0 - mirrored[:, 126:]

        return mirrored.astype(np.float32)

    @staticmethod
    def add_gaussian_noise(seq: np.ndarray, std: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to all keypoint values.

        Args:
            seq: Array of shape (window_size, num_keypoints).
            std: Standard deviation of the noise distribution.

        Returns:
            Noisy array clipped to [0, 1], same shape.
        """
        noise = np.random.normal(0.0, std, seq.shape).astype(np.float32)
        return np.clip(seq + noise, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def random_apply(
        seq: np.ndarray,
        p_warp: float = 0.5,
        p_mirror: float = 0.5,
        p_noise: float = 0.7,
    ) -> np.ndarray:
        """Apply each augmentation independently with given probabilities.

        Args:
            seq: Array of shape (window_size, num_keypoints).
            p_warp: Probability of applying time_warp.
            p_mirror: Probability of applying mirror_horizontal.
            p_noise: Probability of applying add_gaussian_noise.

        Returns:
            Augmented array of the same shape (30, 130).
        """
        if np.random.random() < p_warp:
            seq = KeypointAugmentation.time_warp(seq)
        if np.random.random() < p_mirror:
            seq = KeypointAugmentation.mirror_horizontal(seq)
        if np.random.random() < p_noise:
            seq = KeypointAugmentation.add_gaussian_noise(seq)
        return seq
