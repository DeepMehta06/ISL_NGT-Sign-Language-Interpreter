"""Tests for KeypointExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.config import settings
from backend.pose.extractor import KeypointExtractor


@pytest.fixture(scope="module")
def extractor() -> KeypointExtractor:
    ext = KeypointExtractor(static_image_mode=True)
    yield ext
    ext.close()


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame() -> np.ndarray:
    return np.full((480, 640, 3), 255, dtype=np.uint8)


class TestKeypointExtractorOutput:
    def test_output_shape(self, extractor: KeypointExtractor, blank_frame: np.ndarray) -> None:
        keypoints = extractor.extract(blank_frame)
        assert keypoints.shape == (settings.num_keypoints,)

    def test_output_dtype(self, extractor: KeypointExtractor, blank_frame: np.ndarray) -> None:
        keypoints = extractor.extract(blank_frame)
        assert keypoints.dtype == np.float32

    def test_zero_fill_no_hands(self, extractor: KeypointExtractor, blank_frame: np.ndarray) -> None:
        keypoints = extractor.extract(blank_frame)
        assert np.all(keypoints[:126] == 0.0), "Hand keypoints should be zero when no hands detected."

    def test_output_is_new_array(self, extractor: KeypointExtractor, blank_frame: np.ndarray) -> None:
        k1 = extractor.extract(blank_frame)
        k2 = extractor.extract(blank_frame)
        assert k1 is not k2

    def test_shape_consistent_across_frames(self, extractor: KeypointExtractor) -> None:
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        shapes = [extractor.extract(f).shape for f in frames]
        assert all(s == (settings.num_keypoints,) for s in shapes)

    def test_shape_consistent_different_resolutions(self, extractor: KeypointExtractor) -> None:
        for h, w in [(240, 320), (480, 640), (720, 1280)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            kp = extractor.extract(frame)
            assert kp.shape == (settings.num_keypoints,)


class TestKeypointExtractorValidation:
    def test_raises_on_empty_frame(self, extractor: KeypointExtractor) -> None:
        with pytest.raises(ValueError, match="empty"):
            extractor.extract(np.array([]))

    def test_raises_on_none_frame(self, extractor: KeypointExtractor) -> None:
        with pytest.raises(ValueError, match="empty"):
            extractor.extract(None)

    def test_raises_on_grayscale_frame(self, extractor: KeypointExtractor) -> None:
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((480, 640), dtype=np.uint8))

    def test_raises_on_4channel_frame(self, extractor: KeypointExtractor) -> None:
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((480, 640, 4), dtype=np.uint8))


class TestKeypointExtractorContextManager:
    def test_context_manager(self, blank_frame: np.ndarray) -> None:
        with KeypointExtractor(static_image_mode=True) as ext:
            kp = ext.extract(blank_frame)
        assert kp.shape == (settings.num_keypoints,)

    def test_extract_with_results_shape(self, extractor: KeypointExtractor, blank_frame: np.ndarray) -> None:
        kp, results = extractor.extract_with_results(blank_frame)
        assert kp.shape == (settings.num_keypoints,)
        assert results is not None


class TestKeypointLayout:
    def test_keypoint_count_decomposition(self) -> None:
        left_hand = settings.num_hand_landmarks * settings.num_hand_coords
        right_hand = settings.num_hand_landmarks * settings.num_hand_coords
        pose_ctx = settings.num_pose_context_landmarks
        assert left_hand + right_hand + pose_ctx == settings.num_keypoints

    def test_pose_context_indices_length(self) -> None:
        assert len(settings.pose_context_indices) == settings.num_pose_context_landmarks
