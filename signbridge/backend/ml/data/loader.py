"""Dataset loaders for all four SignBridge datasets.

Handles:
- INCLUDE (ISL isolated signs, zipped categories)
- ISLVT (ISL sentence videos + Excel gloss)
- Indian Sign Language_Dataset (ISL.zip)
- NGT_HoReCo_1.2 (NGT continuous signing videos + Excel annotation)
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.pose.buffer import FrameBuffer
from backend.pose.extractor import KeypointExtractor

logger = get_logger(__name__)


class DatasetLoader:
    """Processes raw dataset videos into (N_sequences, 30, 130) numpy arrays.

    Args:
        extractor: Optional pre-constructed KeypointExtractor. One is
            created internally if not provided.
    """

    def __init__(self, extractor: Optional[KeypointExtractor] = None) -> None:
        self._extractor = extractor or KeypointExtractor(static_image_mode=True)
        self._owns_extractor = extractor is None

    def close(self) -> None:
        """Release the extractor if it was created internally."""
        if self._owns_extractor:
            self._extractor.close()

    def __enter__(self) -> "DatasetLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def process_include(self) -> tuple[np.ndarray, np.ndarray]:
        """Process the INCLUDE ISL isolated-sign dataset.

        Extracts each category zip one at a time, runs pose extraction on all
        videos inside, deletes the extracted folder, then moves to the next zip.
        Only one category occupies disk at a time to minimise peak usage.

        Expected structure after extraction:
            _include_tmp/{sign_label}/{video}.mp4

        Saves:
            data/processed/isl_sequences_include.npy  shape (N, 30, 130)
            data/processed/isl_labels_include.npy     shape (N,)

        Returns:
            Tuple of (sequences, labels).

        Raises:
            FileNotFoundError: If INCLUDE directory is missing or has no zips.
            RuntimeError: If no sequences could be extracted from any zip.
        """
        dataset_dir = settings.include_dataset_dir
        if not dataset_dir.exists():
            raise FileNotFoundError(f"INCLUDE dataset not found at {dataset_dir}")

        zip_files = sorted(dataset_dir.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(
                f"No zip files in {dataset_dir}. "
                "Download INCLUDE category zips and place them there."
            )

        logger.info(f"INCLUDE: {len(zip_files)} zips — processing one at a time.")
        all_sequences: list[np.ndarray] = []
        all_labels: list[str] = []
        extract_root = settings.processed_data_dir / "_include_tmp"

        for i, zip_path in enumerate(zip_files, 1):
            logger.info(f"[{i}/{len(zip_files)}] Extracting {zip_path.name} ...")
            extract_root.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)

            seqs, labs = self._process_video_tree(
                root=extract_root,
                dataset_name=zip_path.stem,
                extensions=(".mp4", ".avi"),
                label_depth=1,
            )
            all_sequences.extend(seqs)
            all_labels.extend(labs.tolist())

            shutil.rmtree(extract_root)
            logger.info(f"Deleted extracted folder for {zip_path.name}.")

        if not all_sequences:
            raise RuntimeError("INCLUDE: no sequences extracted. Check video paths.")

        sequences = np.stack(all_sequences, axis=0).astype(np.float32)
        labels = np.array(all_labels)

        self._save_arrays(
            sequences=sequences,
            labels=labels,
            seq_path=settings.processed_data_dir / "isl_sequences_include.npy",
            label_path=settings.processed_data_dir / "isl_labels_include.npy",
        )
        self._print_stats("INCLUDE", labels)
        return sequences, labels

    def process_include_from_extracted(self, extracted_root: Path) -> tuple[np.ndarray, np.ndarray]:
        """Process an already-extracted INCLUDE directory — no zips required.

        Use when zips have been extracted to disk beforehand. Walks the full
        category directory tree, extracts pose sequences, saves .npy output.

        Args:
            extracted_root: Root folder containing INCLUDE category subdirectories.

        Returns:
            Tuple of (sequences, labels).

        Raises:
            FileNotFoundError: If extracted_root does not exist.
            RuntimeError: If no sequences could be extracted.
        """
        if not extracted_root.exists():
            raise FileNotFoundError(f"Extracted directory not found: {extracted_root}")

        sequences, labels = self._process_video_tree(
            root=extracted_root,
            dataset_name="INCLUDE",
            extensions=(".mp4", ".avi"),
            label_depth=1,
        )

        if len(sequences) == 0:
            raise RuntimeError("INCLUDE: no sequences extracted. Check video paths.")

        out_seq = settings.processed_data_dir / "isl_sequences_include.npy"
        out_lab = settings.processed_data_dir / "isl_labels_include.npy"
        self._save_arrays(sequences=sequences, labels=labels, seq_path=out_seq, label_path=out_lab)
        self._print_stats("INCLUDE", labels)
        return sequences, labels

    def process_islvt(self) -> tuple[np.ndarray, np.ndarray]:
        """Process the ISLVT ISL sentence video dataset.

        Videos are stored flat with filenames encoding the sentence gloss.
        Label for each video is the numeric prefix (sentence ID).

        Expected structure:
            ISLVT/{sentence_id} {words...}.MOV
            ISLVT/marathi-English_ISLgloss1.xlsx

        Saves:
            data/processed/isl_sequences_islvt.npy
            data/processed/isl_labels_islvt.npy

        Returns:
            Tuple of (sequences, labels).

        Raises:
            FileNotFoundError: If the ISLVT dataset directory is missing.
        """
        dataset_dir = settings.islvt_dataset_dir
        if not dataset_dir.exists():
            raise FileNotFoundError(f"ISLVT dataset not found at {dataset_dir}")

        video_files = sorted(
            f for f in dataset_dir.iterdir()
            if f.suffix.upper() in (".MOV", ".MP4", ".AVI")
        )
        logger.info(f"ISLVT: found {len(video_files)} video files.")

        sequences, labels = [], []
        for video_path in tqdm(video_files, desc="Processing ISLVT"):
            sentence_id = video_path.stem.split()[0]
            seqs = self._extract_sequences_from_video(video_path)
            for seq in seqs:
                sequences.append(seq)
                labels.append(sentence_id)

        if not sequences:
            logger.warning("ISLVT: no sequences extracted.")
            return np.empty((0, settings.window_size, settings.num_keypoints)), np.array([])

        seq_arr = np.stack(sequences, axis=0).astype(np.float32)
        label_arr = np.array(labels)

        self._save_arrays(
            sequences=seq_arr,
            labels=label_arr,
            seq_path=settings.processed_data_dir / "isl_sequences_islvt.npy",
            label_path=settings.processed_data_dir / "isl_labels_islvt.npy",
        )
        self._print_stats("ISLVT", label_arr)
        return seq_arr, label_arr

    def process_isl_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Process the Indian Sign Language_Dataset (ISL.zip).

        Extracts ISL.zip and processes every video found inside.
        Label is inferred from the parent folder name after extraction.

        Saves:
            data/processed/isl_sequences_isl_dataset.npy
            data/processed/isl_labels_isl_dataset.npy

        Returns:
            Tuple of (sequences, labels).

        Raises:
            FileNotFoundError: If the dataset directory or ISL.zip is missing.
        """
        dataset_dir = settings.isl_dataset_dir
        zip_path = dataset_dir / "ISL.zip"

        if not zip_path.exists():
            raise FileNotFoundError(f"ISL.zip not found at {zip_path}")

        extract_root = settings.processed_data_dir / "_isl_extracted"
        extract_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

        sequences, labels = self._process_video_tree(
            root=extract_root,
            dataset_name="ISL_Dataset",
            extensions=(".mp4", ".avi", ".mov"),
            label_depth=1,
        )

        self._save_arrays(
            sequences=sequences,
            labels=labels,
            seq_path=settings.processed_data_dir / "isl_sequences_isl_dataset.npy",
            label_path=settings.processed_data_dir / "isl_labels_isl_dataset.npy",
        )
        self._print_stats("ISL_Dataset", labels)
        return sequences, labels

    def process_ngt_horeco(self) -> tuple[np.ndarray, np.ndarray]:
        """Process the NGT HoReCo 1.2 continuous signing dataset.

        Videos are stored flat in NGT_HoReCo_1.2/Videos/. Each video
        gets its own label derived from the filename stem.
        The companion Excel (Multilingual-HoReCo.xlsx) is parsed in Phase 3.

        Saves:
            data/processed/ngt_sequences.npy
            data/processed/ngt_labels.npy

        Returns:
            Tuple of (sequences, labels).

        Raises:
            FileNotFoundError: If the NGT dataset directory is missing.
        """
        videos_dir = settings.ngt_dataset_dir / "Videos"
        if not videos_dir.exists():
            raise FileNotFoundError(f"NGT Videos directory not found at {videos_dir}")

        video_files = sorted(
            f for f in videos_dir.iterdir()
            if f.suffix.lower() in (".mp4", ".mov", ".avi")
        )
        logger.info(f"NGT HoReCo: found {len(video_files)} video files.")

        sequences, labels = [], []
        for video_path in tqdm(video_files, desc="Processing NGT HoReCo"):
            seqs = self._extract_sequences_from_video(video_path)
            for seq in seqs:
                sequences.append(seq)
                labels.append(video_path.stem)

        if not sequences:
            logger.warning("NGT HoReCo: no sequences extracted.")
            return np.empty((0, settings.window_size, settings.num_keypoints)), np.array([])

        seq_arr = np.stack(sequences, axis=0).astype(np.float32)
        label_arr = np.array(labels)

        self._save_arrays(
            sequences=seq_arr,
            labels=label_arr,
            seq_path=settings.processed_data_dir / "ngt_sequences.npy",
            label_path=settings.processed_data_dir / "ngt_labels.npy",
        )
        self._print_stats("NGT HoReCo", label_arr)
        return seq_arr, label_arr

    def _process_video_tree(
        self,
        root: Path,
        dataset_name: str,
        extensions: tuple[str, ...],
        label_depth: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Walk a directory tree and extract sequences from all matching videos.

        Args:
            root: Root directory to walk.
            dataset_name: Name used in log messages.
            extensions: Lowercase file extensions to process.
            label_depth: How many folder levels below root to read as label.

        Returns:
            Tuple of (sequences array shape (N, 30, 130), labels array shape (N,)).
        """
        video_paths = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]
        logger.info(f"{dataset_name}: found {len(video_paths)} videos to process.")

        sequences, labels = [], []
        for video_path in tqdm(video_paths, desc=f"Processing {dataset_name}"):
            parts = video_path.relative_to(root).parts
            label = parts[-1 - label_depth] if len(parts) > label_depth else video_path.stem
            seqs = self._extract_sequences_from_video(video_path)
            for seq in seqs:
                sequences.append(seq)
                labels.append(label)

        if not sequences:
            return np.empty((0, settings.window_size, settings.num_keypoints)), np.array([])

        return np.stack(sequences, axis=0).astype(np.float32), np.array(labels)

    def _extract_sequences_from_video(self, video_path: Path) -> list[np.ndarray]:
        """Extract all 30-frame windows from a single video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of arrays of shape (30, 130). Empty if video cannot be opened
            or has fewer than 30 frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return []

        buffer = FrameBuffer()
        windows: list[np.ndarray] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    keypoints = self._extractor.extract(frame)
                    window = buffer.add(keypoints)
                    if window is not None:
                        windows.append(window)
                except ValueError as exc:
                    logger.warning(f"Frame skipped in {video_path.name}: {exc}")
        finally:
            cap.release()

        return windows

    @staticmethod
    def _save_arrays(
        sequences: np.ndarray,
        labels: np.ndarray,
        seq_path: Path,
        label_path: Path,
    ) -> None:
        """Save sequences and labels as numpy .npy files.

        Args:
            sequences: Float32 array of shape (N, 30, 130).
            labels: String array of shape (N,).
            seq_path: Output path for sequences.
            label_path: Output path for labels.
        """
        np.save(seq_path, sequences)
        np.save(label_path, labels)
        logger.info(f"Saved sequences: {seq_path} {sequences.shape}")
        logger.info(f"Saved labels:    {label_path} {labels.shape}")

    @staticmethod
    def _print_stats(dataset_name: str, labels: np.ndarray) -> None:
        """Log dataset statistics after processing.

        Args:
            dataset_name: Human-readable name for log messages.
            labels: Label array of shape (N,).
        """
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(
            f"\n{'='*50}\n"
            f"Dataset:         {dataset_name}\n"
            f"Total sequences: {len(labels)}\n"
            f"Unique classes:  {len(unique)}\n"
            f"Min per class:   {counts.min() if len(counts) else 0}\n"
            f"Max per class:   {counts.max() if len(counts) else 0}\n"
            f"Mean per class:  {counts.mean():.1f if len(counts) else 0}\n"
            f"{'='*50}"
        )
