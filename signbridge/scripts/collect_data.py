"""Data collection CLI for SignBridge self-recorded signs.

Usage:
    python scripts/collect_data.py --sign "hello" --samples 50 --language ISL
    python scripts/collect_data.py --sign "water" --samples 30 --language NGT

Records N samples of a specified sign from the webcam, shows live skeleton
overlay, and saves each sample as a (30, 130) numpy array.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.pose.extractor import KeypointExtractor
from backend.pose.buffer import FrameBuffer
from backend.pose.visualizer import SkeletonVisualiser

logger = get_logger(__name__)

COUNTDOWN_SECONDS = 3
WINDOW_TITLE = "SignBridge — Data Collection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect self-recorded sign samples.")
    parser.add_argument("--sign", required=True, help="Sign label to record (e.g. 'hello').")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to record.")
    parser.add_argument(
        "--language",
        choices=["ISL", "NGT"],
        default="ISL",
        help="Target sign language.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    return parser.parse_args()


def countdown(cap: cv2.VideoCapture, vis: SkeletonVisualiser, language: str, sign: str) -> None:
    """Display a live countdown overlay before recording starts."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        remaining = COUNTDOWN_SECONDS - int(elapsed)
        if remaining <= 0:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            f"Get ready: {sign}  ({remaining}s)",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (26, 26, 46),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def collect_sample(
    cap: cv2.VideoCapture,
    extractor: KeypointExtractor,
    vis: SkeletonVisualiser,
    language: str,
) -> np.ndarray | None:
    """Collect one 30-frame sample from the webcam.

    Returns a (30, 130) array or None if collection failed.
    """
    buffer = FrameBuffer()
    frames_recorded = 0

    while frames_recorded < settings.window_size:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        try:
            keypoints, results = extractor.extract_with_results(frame)
        except ValueError:
            continue

        window = buffer.add(keypoints)
        annotated = vis.draw(frame, results, language=language)

        progress = frames_recorded / settings.window_size
        h, w = annotated.shape[:2]
        bar_w = int((w - 8) * progress)
        accent = (178, 207, 62) if language == "ISL" else (250, 139, 167)
        cv2.rectangle(annotated, (4, h - 10), (4 + bar_w, h - 4), accent, -1, cv2.LINE_AA)

        cv2.putText(
            annotated,
            f"Recording... {frames_recorded}/{settings.window_size}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (26, 26, 46),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_TITLE, annotated)
        cv2.waitKey(1)

        frames_recorded += 1

        if window is not None:
            return window

    if buffer.current_length >= settings.window_size:
        frames = list(buffer._buffer)[:settings.window_size]
        return np.stack(frames, axis=0)

    return None


def main() -> None:
    args = parse_args()

    output_dir = settings.self_recorded_dir / args.language / args.sign
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("sequence_*.npy"))
    start_idx = len(existing)
    logger.info(f"Saving to: {output_dir} | Starting at index {start_idx}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {args.camera}.")
        return

    extractor = KeypointExtractor(static_image_mode=False)
    vis = SkeletonVisualiser()
    collected = 0

    try:
        while collected < args.samples:
            logger.info(f"Sample {collected + 1}/{args.samples} — sign: '{args.sign}'")
            countdown(cap, vis, args.language, args.sign)

            sample = collect_sample(cap, extractor, vis, args.language)
            if sample is None:
                logger.warning("Sample collection failed — retrying.")
                continue

            save_path = output_dir / f"sequence_{start_idx + collected:04d}.npy"
            np.save(save_path, sample)
            logger.info(f"Saved {save_path} | shape={sample.shape}")
            collected += 1

            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame,
                    f"Saved {collected}/{args.samples}! Press any key for next.",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (52, 201, 62),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_TITLE, frame)
                cv2.waitKey(800)
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()

    logger.info(f"Collection complete. {collected} samples saved to {output_dir}")


if __name__ == "__main__":
    main()
