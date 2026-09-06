"""Skeleton visualiser for SignBridge pose debugging.

Draws MediaPipe hand and pose landmarks on a BGR frame. ISL = teal, NGT = violet.
For debugging and data collection only — not part of the inference pipeline.
"""

from __future__ import annotations

import numpy as np
import cv2

from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# BGR colors matching DESIGN.md
_COLOR_ISL = (178, 207, 62)     # teal  #3ECFB2
_COLOR_NGT = (250, 139, 167)    # violet #A78BFA
_COLOR_POSE = (239, 141, 91)    # blue  #5B8DEF

# MediaPipe hand connections (finger pairs, 0-indexed)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class SkeletonVisualiser:
    """Draws keypoints on BGR frames for debugging.

    Args:
        hand_landmark_radius: Pixel radius for hand landmark dots.
        connection_thickness: Pixel thickness for connecting lines.
        pose_landmark_radius: Pixel radius for pose context dots.
    """

    def __init__(
        self,
        hand_landmark_radius: int = 5,
        connection_thickness: int = 2,
        pose_landmark_radius: int = 3,
    ) -> None:
        self._hand_r = hand_landmark_radius
        self._conn_t = connection_thickness
        self._pose_r = pose_landmark_radius

    def draw(
        self,
        frame: np.ndarray,
        results: object,
        language: str = "ISL",
        show_confidence: float | None = None,
    ) -> np.ndarray:
        """Draw skeleton overlay on a BGR frame.

        Args:
            frame: BGR image array (H, W, 3).
            results: HolisticLandmarkerResult from KeypointExtractor.extract_with_results().
            language: "ISL" (teal) or "NGT" (violet).
            show_confidence: Optional [0,1] float to draw as a bottom progress bar.

        Returns:
            New BGR array with skeleton drawn.

        Raises:
            ValueError: If language is not "ISL" or "NGT".
        """
        if language not in ("ISL", "NGT"):
            raise ValueError(f"language must be 'ISL' or 'NGT', got '{language}'.")

        annotated = frame.copy()
        h, w = annotated.shape[:2]
        color = _COLOR_ISL if language == "ISL" else _COLOR_NGT

        self._draw_hand_landmarks(annotated, results.left_hand_landmarks, w, h, color)
        self._draw_hand_landmarks(annotated, results.right_hand_landmarks, w, h, color)
        self._draw_pose_context(annotated, results.pose_landmarks, w, h)
        self._draw_badge(annotated, language, color)

        if show_confidence is not None:
            self._draw_confidence_bar(annotated, show_confidence, color, w, h)

        return annotated

    def draw_from_keypoints(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        language: str = "ISL",
    ) -> np.ndarray:
        """Draw skeleton from a (130,) keypoint vector.

        Args:
            frame: BGR image array (H, W, 3).
            keypoints: Float32 array of shape (130,).
            language: "ISL" or "NGT".

        Returns:
            New BGR array with landmarks drawn.

        Raises:
            ValueError: If keypoints does not have shape (130,).
        """
        if keypoints.shape != (settings.num_keypoints,):
            raise ValueError(f"Expected shape ({settings.num_keypoints},), got {keypoints.shape}.")

        annotated = frame.copy()
        h, w = annotated.shape[:2]
        color = _COLOR_ISL if language == "ISL" else _COLOR_NGT

        left = keypoints[:63].reshape(21, 3)
        right = keypoints[63:126].reshape(21, 3)
        self._draw_landmarks_array(annotated, left, w, h, color)
        self._draw_connections_array(annotated, left, w, h, color)
        self._draw_landmarks_array(annotated, right, w, h, color)
        self._draw_connections_array(annotated, right, w, h, color)
        self._draw_badge(annotated, language, color)
        return annotated

    def _draw_hand_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list | None,
        w: int,
        h: int,
        color: tuple[int, int, int],
    ) -> None:
        if not landmarks:
            return
        for s, e in _HAND_CONNECTIONS:
            sx, sy = int(landmarks[s].x * w), int(landmarks[s].y * h)
            ex, ey = int(landmarks[e].x * w), int(landmarks[e].y * h)
            cv2.line(frame, (sx, sy), (ex, ey), color, self._conn_t, cv2.LINE_AA)
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), self._hand_r, color, -1, cv2.LINE_AA)

    def _draw_pose_context(
        self,
        frame: np.ndarray,
        pose_landmarks: list | None,
        w: int,
        h: int,
    ) -> None:
        if not pose_landmarks:
            return
        for idx in settings.pose_context_indices:
            lm = pose_landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), self._pose_r, _COLOR_POSE, -1, cv2.LINE_AA)

    def _draw_landmarks_array(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        w: int,
        h: int,
        color: tuple[int, int, int],
    ) -> None:
        for lm in landmarks:
            if lm[0] == 0.0 and lm[1] == 0.0:
                continue
            cv2.circle(frame, (int(lm[0] * w), int(lm[1] * h)), self._hand_r, color, -1, cv2.LINE_AA)

    def _draw_connections_array(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        w: int,
        h: int,
        color: tuple[int, int, int],
    ) -> None:
        for s, e in _HAND_CONNECTIONS:
            if (landmarks[s][0] == 0.0 and landmarks[s][1] == 0.0) or \
               (landmarks[e][0] == 0.0 and landmarks[e][1] == 0.0):
                continue
            cv2.line(
                frame,
                (int(landmarks[s][0] * w), int(landmarks[s][1] * h)),
                (int(landmarks[e][0] * w), int(landmarks[e][1] * h)),
                color, self._conn_t, cv2.LINE_AA,
            )

    def _draw_badge(self, frame: np.ndarray, language: str, color: tuple[int, int, int]) -> None:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(language, font, 0.6, 2)
        pad = 8
        x1, y1 = w - tw - 2 * pad - 12, 12
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 2 * pad, y1 + th + 2 * pad), color, -1, cv2.LINE_AA)
        cv2.putText(frame, language, (x1 + pad, y1 + th + pad), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_confidence_bar(
        self,
        frame: np.ndarray,
        confidence: float,
        color: tuple[int, int, int],
        w: int,
        h: int,
    ) -> None:
        bar_y = h - 10
        cv2.rectangle(frame, (4, bar_y), (w - 4, bar_y + 6), (240, 237, 232), -1, cv2.LINE_AA)
        fill_w = int((w - 8) * max(0.0, min(1.0, confidence)))
        if fill_w > 0:
            cv2.rectangle(frame, (4, bar_y), (4 + fill_w, bar_y + 6), color, -1, cv2.LINE_AA)
