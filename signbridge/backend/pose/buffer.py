"""Sliding-window frame buffer for the SignBridge pose pipeline.

This module implements the second layer of the inference pipeline. It
accumulates per-frame keypoint vectors and emits fixed-size windows
for the ML classifier.

Window behaviour:
    - window_size: 30 frames
    - stride:      15 frames (50% overlap)

When the buffer reaches 30 frames it yields a (30, 130) numpy array.
The window then slides forward by 15 frames — the last 15 frames are
kept and the next 15 are appended before the next emission.

Example::

    buffer = FrameBuffer()
    for keypoints in keypoint_stream:
        window = buffer.add(keypoints)
        if window is not None:
            prediction = model(window)
    buffer.reset()
"""

from __future__ import annotations

from collections import deque

import numpy as np

from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class FrameBuffer:
    """Sliding-window accumulator for keypoint frame sequences.

    Maintains an internal deque of (num_keypoints,) vectors. Once the
    deque reaches ``window_size`` frames, it packages them as a numpy
    array of shape (window_size, num_keypoints) and advances by
    ``stride`` frames.

    Attributes:
        window_size: Number of frames per window (default 30).
        stride: Number of frames to advance after each window (default 15).
        num_keypoints: Expected keypoint vector length (default 130).

    Example::

        buf = FrameBuffer()
        for frame_keypoints in keypoint_stream:
            window = buf.add(frame_keypoints)
            if window is not None:
                run_inference(window)
    """

    def __init__(
        self,
        window_size: int | None = None,
        stride: int | None = None,
        num_keypoints: int | None = None,
    ) -> None:
        """Initialise the frame buffer with configurable dimensions.

        Args:
            window_size: Number of frames per window. Defaults to
                ``settings.window_size`` (30).
            stride: Number of frames to advance the window after emission.
                Defaults to ``settings.stride`` (15). Must be <= window_size.
            num_keypoints: Expected length of each keypoint vector.
                Defaults to ``settings.num_keypoints`` (130).

        Raises:
            ValueError: If stride > window_size.
        """
        self.window_size: int = window_size or settings.window_size
        self.stride: int = stride or settings.stride
        self.num_keypoints: int = num_keypoints or settings.num_keypoints

        if self.stride > self.window_size:
            raise ValueError(
                f"stride ({self.stride}) must be <= window_size ({self.window_size})."
            )

        # deque with maxlen ensures automatic eviction; we manage manually.
        self._buffer: deque[np.ndarray] = deque()
        self._frames_since_last_emit: int = 0

        logger.debug(
            f"FrameBuffer initialised | window={self.window_size} | "
            f"stride={self.stride} | keypoints={self.num_keypoints}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, keypoints: np.ndarray) -> np.ndarray | None:
        """Add a single frame's keypoints and return a window if ready.

        Args:
            keypoints: Float32 array of shape (num_keypoints,) for one frame.

        Returns:
            A numpy array of shape (window_size, num_keypoints) when the
            buffer has accumulated enough frames, otherwise None.

        Raises:
            ValueError: If ``keypoints`` has an unexpected shape.
        """
        if keypoints.shape != (self.num_keypoints,):
            raise ValueError(
                f"Expected keypoints of shape ({self.num_keypoints},), "
                f"got {keypoints.shape}."
            )

        self._buffer.append(keypoints.astype(np.float32))
        self._frames_since_last_emit += 1

        if len(self._buffer) >= self.window_size:
            return self._emit()

        return None

    def reset(self) -> None:
        """Clear the buffer completely.

        Call this when a new signing session starts or after a long pause
        to avoid cross-sentence contamination.
        """
        self._buffer.clear()
        self._frames_since_last_emit = 0
        logger.debug("FrameBuffer reset.")

    @property
    def current_length(self) -> int:
        """Return the number of frames currently in the buffer.

        Returns:
            Integer count of buffered frames.
        """
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Return True if the buffer has reached window_size frames.

        Returns:
            True when ``current_length >= window_size``.
        """
        return len(self._buffer) >= self.window_size

    # ── Private helpers ────────────────────────────────────────────────────────

    def _emit(self) -> np.ndarray:
        """Package the current window and advance by stride.

        Takes the first ``window_size`` frames from the buffer as a numpy
        array, removes the leading ``stride`` frames, and resets the
        since-last-emit counter.

        Returns:
            Float32 numpy array of shape (window_size, num_keypoints).
        """
        # Take exactly window_size frames from the front of the deque.
        window_frames = list(self._buffer)[:self.window_size]
        window = np.stack(window_frames, axis=0)  # (window_size, num_keypoints)

        # Advance by stride: remove leading `stride` frames.
        for _ in range(self.stride):
            if self._buffer:
                self._buffer.popleft()

        self._frames_since_last_emit = 0

        logger.debug(
            f"Window emitted | shape={window.shape} | "
            f"buffer_remaining={len(self._buffer)}"
        )
        return window
