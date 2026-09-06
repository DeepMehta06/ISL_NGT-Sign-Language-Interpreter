"""SignBridge global configuration.

All constants, paths, and hyperparameters are defined here using
pydantic-settings. Values can be overridden via environment variables
or a .env file at the project root.

Never hardcode magic numbers anywhere else in the codebase.
Always import from this module.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration for SignBridge.

    All fields can be overridden via environment variables (uppercase)
    or a .env file in the project root.

    Attributes:
        log_level: Loguru log level string (DEBUG, INFO, WARNING, ERROR).
        device: PyTorch device string ('cpu' or 'cuda').

        # Pose extraction
        num_keypoints: Total keypoints extracted per frame (130).
        num_hand_landmarks: MediaPipe hand landmarks per hand (21).
        num_hand_coords: XYZ coordinates per landmark (3).
        num_pose_context_landmarks: Shoulder + wrist landmarks used (4).
        window_size: Sliding window frame count for the frame buffer (30).
        stride: Frame stride for the sliding window (15).

        # Inference
        confidence_threshold: Minimum confidence to consider a sign (0.85).
        pause_timeout_seconds: Silence duration to finalise a sentence (1.5).
        commit_streak: Consecutive windows needed to commit a sign (2).

        # Paths
        project_root: Absolute path to the signbridge/ project folder.
        data_dir: Root directory containing all datasets (raw).
        processed_data_dir: Output directory for preprocessed numpy arrays.
        self_recorded_dir: Output directory for self-recorded signs.
        model_dir: Directory for trained .pt model weights.

        # Dataset-specific paths
        include_dataset_dir: Path to the INCLUDE ISL dataset root.
        islvt_dataset_dir: Path to ISLVT sentence dataset root.
        isl_dataset_dir: Path to Indian Sign Language_Dataset root.
        ngt_dataset_dir: Path to NGT_HoReCo_1.2 dataset root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Logging & hardware ───────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Loguru log level.")
    device: str = Field(default="cpu", description="PyTorch device.")

    # ── Pose extraction constants ────────────────────────────────────────────
    num_keypoints: int = Field(
        default=130,
        description="Total float32 values extracted per frame.",
    )
    num_hand_landmarks: int = Field(
        default=21,
        description="MediaPipe landmarks per hand.",
    )
    num_hand_coords: int = Field(
        default=3,
        description="XYZ coordinates per landmark.",
    )
    num_pose_context_landmarks: int = Field(
        default=4,
        description="Number of pose landmarks used for context (x-only).",
    )

    # ── Frame buffer ─────────────────────────────────────────────────────────
    window_size: int = Field(
        default=30,
        description="Sliding window size in frames.",
    )
    stride: int = Field(
        default=15,
        description="Sliding window stride in frames (50% overlap).",
    )

    # ── Inference ────────────────────────────────────────────────────────────
    confidence_threshold: float = Field(
        default=0.85,
        description="Minimum model confidence to accept a sign prediction.",
    )
    pause_timeout_seconds: float = Field(
        default=1.5,
        description="Seconds of silence before a sentence is finalised.",
    )
    commit_streak: int = Field(
        default=2,
        description="Consecutive windows above threshold to commit a word.",
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    project_root: Path = Field(
        default=Path(__file__).resolve().parents[2],
        description="Absolute path to the signbridge/ root directory.",
    )
    data_dir: Path = Field(
        default=Path(__file__).resolve().parents[3] / "Datasets",
        description="Root directory for all raw datasets.",
    )
    processed_data_dir: Path = Field(
        default=Path(__file__).resolve().parents[2] / "data" / "processed",
        description="Output directory for preprocessed numpy arrays.",
    )
    self_recorded_dir: Path = Field(
        default=Path(__file__).resolve().parents[2] / "data" / "self_recorded",
        description="Directory for self-recorded sign sequences.",
    )
    model_dir: Path = Field(
        default=Path(__file__).resolve().parents[2] / "backend" / "models",
        description="Directory for trained PyTorch model weights.",
    )

    # ── Dataset paths ────────────────────────────────────────────────────────
    include_dataset_dir: Path = Field(
        default=Path(__file__).resolve().parents[3] / "Datasets" / "INCLUDE",
        description="Path to the INCLUDE ISL isolated-sign dataset.",
    )
    islvt_dataset_dir: Path = Field(
        default=Path(__file__).resolve().parents[3]
        / "Datasets"
        / "Indian Sign Language Video and Text dataset for sentences (ISLVT)",
        description="Path to ISLVT ISL sentence video dataset.",
    )
    isl_dataset_dir: Path = Field(
        default=Path(__file__).resolve().parents[3]
        / "Datasets"
        / "Indian Sign Language_Dataset",
        description="Path to Indian Sign Language_Dataset (contains ISL.zip).",
    )
    ngt_dataset_dir: Path = Field(
        default=Path(__file__).resolve().parents[3] / "Datasets" / "NGT_HoReCo_1.2",
        description="Path to NGT HoReCo 1.2 dataset.",
    )

    # ── MediaPipe pose landmark indices ─────────────────────────────────────
    # Indices within MediaPipe's full 33-landmark pose set.
    # Used to extract only the 4 context values (shoulders + wrists).
    pose_context_indices: list[int] = Field(
        default=[11, 12, 15, 16],  # left shoulder, right shoulder, left wrist, right wrist
        description="MediaPipe pose landmark indices for context extraction.",
    )

    def model_post_init(self, __context: object) -> None:
        """Ensure all output directories exist after model initialisation."""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.self_recorded_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere:
#   from backend.core.config import settings
settings = Settings()
