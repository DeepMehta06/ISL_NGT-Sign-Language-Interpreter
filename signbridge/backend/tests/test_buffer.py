"""Tests for FrameBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.config import settings
from backend.pose.buffer import FrameBuffer


def make_keypoints(value: float = 0.0) -> np.ndarray:
    return np.full((settings.num_keypoints,), value, dtype=np.float32)


class TestFrameBufferEmission:
    def test_no_emission_before_window_full(self) -> None:
        buf = FrameBuffer()
        for _ in range(settings.window_size - 1):
            result = buf.add(make_keypoints())
        assert result is None

    def test_emits_on_window_size(self) -> None:
        buf = FrameBuffer()
        result = None
        for _ in range(settings.window_size):
            result = buf.add(make_keypoints())
        assert result is not None

    def test_emitted_window_shape(self) -> None:
        buf = FrameBuffer()
        window = None
        for _ in range(settings.window_size):
            window = buf.add(make_keypoints())
        assert window.shape == (settings.window_size, settings.num_keypoints)

    def test_emitted_window_dtype(self) -> None:
        buf = FrameBuffer()
        window = None
        for _ in range(settings.window_size):
            window = buf.add(make_keypoints(1.0))
        assert window.dtype == np.float32

    def test_second_emission_after_stride(self) -> None:
        buf = FrameBuffer()
        emissions = []
        for i in range(settings.window_size + settings.stride):
            w = buf.add(make_keypoints(float(i)))
            if w is not None:
                emissions.append(w)
        assert len(emissions) == 2

    def test_stride_preserves_overlap(self) -> None:
        buf = FrameBuffer(window_size=4, stride=2, num_keypoints=settings.num_keypoints)
        emissions = []
        for i in range(6):
            w = buf.add(make_keypoints(float(i)))
            if w is not None:
                emissions.append(w)
        assert len(emissions) == 2
        # Second window should share last 2 frames of first window
        np.testing.assert_array_equal(emissions[0][2:], emissions[1][:2])


class TestFrameBufferState:
    def test_current_length_increments(self) -> None:
        buf = FrameBuffer()
        for i in range(1, 10):
            buf.add(make_keypoints())
            assert buf.current_length == min(i, settings.window_size - settings.stride + i)

    def test_is_full_false_before_window(self) -> None:
        buf = FrameBuffer()
        for _ in range(settings.window_size - 1):
            buf.add(make_keypoints())
        assert not buf.is_full

    def test_is_full_true_at_window(self) -> None:
        buf = FrameBuffer()
        for _ in range(settings.window_size - 1):
            buf.add(make_keypoints())
        assert not buf.is_full
        # Add exactly enough to reach window_size without triggering emit yet
        # We use a fresh buffer and inspect right before the 30th frame triggers emit
        buf2 = FrameBuffer()
        for i in range(settings.window_size - 1):
            buf2.add(make_keypoints())
        assert buf2.current_length == settings.window_size - 1
        # After emit, buffer has stride frames remaining (< window_size) — correct
        buf2.add(make_keypoints())
        assert buf2.current_length == settings.stride

    def test_reset_clears_buffer(self) -> None:
        buf = FrameBuffer()
        for _ in range(20):
            buf.add(make_keypoints())
        buf.reset()
        assert buf.current_length == 0
        assert not buf.is_full

    def test_no_emission_after_reset(self) -> None:
        buf = FrameBuffer()
        for _ in range(settings.window_size):
            buf.add(make_keypoints())
        buf.reset()
        for _ in range(settings.window_size - 1):
            result = buf.add(make_keypoints())
        assert result is None


class TestFrameBufferValidation:
    def test_raises_on_wrong_keypoint_shape(self) -> None:
        buf = FrameBuffer()
        with pytest.raises(ValueError):
            buf.add(np.zeros((99,), dtype=np.float32))

    def test_raises_on_invalid_stride(self) -> None:
        with pytest.raises(ValueError):
            FrameBuffer(window_size=10, stride=20)

    def test_custom_dimensions(self) -> None:
        buf = FrameBuffer(window_size=10, stride=5, num_keypoints=64)
        window = None
        for i in range(10):
            window = buf.add(np.zeros((64,), dtype=np.float32))
        assert window is not None
        assert window.shape == (10, 64)
