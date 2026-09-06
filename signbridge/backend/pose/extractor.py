"""MediaPipe Holistic keypoint extractor for SignBridge.

Keypoint layout (130 values total):
    [0:63]    — Left hand: 21 landmarks × 3 (x, y, z)
    [63:126]  — Right hand: 21 landmarks × 3 (x, y, z)
    [126:130] — Pose context: left shoulder, right shoulder,
                left wrist, right wrist (x-coord only)

Missing hands or pose landmarks are zero-filled so output is always (130,).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "backend" / "models" / "holistic_landmarker.task"

VisionRunningMode = mp_vision.RunningMode
HolisticLandmarker = mp_vision.HolisticLandmarker
HolisticLandmarkerOptions = mp_vision.HolisticLandmarkerOptions
BaseOptions = mp_python.BaseOptions


class KeypointExtractor:
    """Extracts 130-dimensional keypoint vectors from BGR video frames.

    Uses MediaPipe HolisticLandmarker (Tasks API). Call ``close()`` or use
    as a context manager when done.

    Args:
        static_image_mode: If True, runs in IMAGE mode (no tracking between
            frames). Use False (VIDEO mode) for continuous video streams.
        min_detection_confidence: Minimum confidence for pose/hand detection.
        model_path: Path to the holistic_landmarker.task model file.

    Example::

        with KeypointExtractor() as extractor:
            for frame in video_frames:
                keypoints = extractor.extract(frame)
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        model_path: Path | None = None,
    ) -> None:
        self._num_keypoints = settings.num_keypoints
        self._num_hand_landmarks = settings.num_hand_landmarks
        self._num_hand_coords = settings.num_hand_coords
        self._pose_context_indices = settings.pose_context_indices

        resolved_model = model_path or _DEFAULT_MODEL_PATH
        if not resolved_model.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {resolved_model}. "
                "Download it with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task "
                f"-OutFile {resolved_model}"
            )

        running_mode = VisionRunningMode.IMAGE if static_image_mode else VisionRunningMode.VIDEO
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=running_mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_hand_landmarks_confidence=min_detection_confidence,
        )
        self._landmarker = HolisticLandmarker.create_from_options(options)
        self._static_mode = static_image_mode
        self._frame_timestamp_ms = 0

        logger.info(
            f"KeypointExtractor initialised | static={static_image_mode} | "
            f"model={resolved_model.name}"
        )

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract a 130-dimensional keypoint vector from a single BGR frame.

        Args:
            frame: BGR image array of shape (H, W, 3).

        Returns:
            Float32 numpy array of shape (130,).

        Raises:
            ValueError: If frame is empty or has wrong shape.
        """
        _, results = self.extract_with_results(frame)
        left_hand = self._extract_hand(results.left_hand_landmarks)
        right_hand = self._extract_hand(results.right_hand_landmarks)
        pose_ctx = self._extract_pose_context(results.pose_landmarks)
        keypoints = np.concatenate([left_hand, right_hand, pose_ctx])

        assert keypoints.shape == (self._num_keypoints,)
        return keypoints

    def extract_with_results(self, frame: np.ndarray) -> tuple[np.ndarray, object]:
        """Extract keypoints and return raw MediaPipe results.

        Args:
            frame: BGR image array of shape (H, W, 3).

        Returns:
            Tuple of (keypoints array shape (130,), HolisticLandmarkerResult).

        Raises:
            ValueError: If frame is empty or has wrong shape.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Received an empty or None frame.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR frame (H, W, 3), got {frame.shape}.")

        rgb = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._static_mode:
            results = self._landmarker.detect(mp_image)
        else:
            self._frame_timestamp_ms += 33  # ~30fps
            results = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        left_hand = self._extract_hand(results.left_hand_landmarks)
        right_hand = self._extract_hand(results.right_hand_landmarks)
        pose_ctx = self._extract_pose_context(results.pose_landmarks)
        keypoints = np.concatenate([left_hand, right_hand, pose_ctx])

        return keypoints, results

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("KeypointExtractor closed.")

    def __enter__(self) -> "KeypointExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _extract_hand(self, hand_landmarks: list | None) -> np.ndarray:
        """Flatten hand landmarks into a (63,) float32 array.

        Args:
            hand_landmarks: List of NormalizedLandmark objects or None.

        Returns:
            Float32 array of shape (63,), zero-filled if None.
        """
        n = self._num_hand_landmarks * self._num_hand_coords
        if not hand_landmarks:
            return np.zeros(n, dtype=np.float32)
        return np.array(
            [coord for lm in hand_landmarks for coord in (lm.x, lm.y, lm.z)],
            dtype=np.float32,
        )

    def _extract_pose_context(self, pose_landmarks: list | None) -> np.ndarray:
        """Extract x-coords of 4 pose context landmarks (shoulders + wrists).

        Args:
            pose_landmarks: List of NormalizedLandmark objects or None.

        Returns:
            Float32 array of shape (4,), zero-filled if None.
        """
        n = settings.num_pose_context_landmarks
        if not pose_landmarks:
            return np.zeros(n, dtype=np.float32)
        return np.array(
            [pose_landmarks[idx].x for idx in self._pose_context_indices],
            dtype=np.float32,
        )
