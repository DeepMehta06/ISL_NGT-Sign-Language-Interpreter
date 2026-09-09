"""Phase 2 runner — executes all Phase 2 steps in order.

Steps:
    1. Pose extraction (skipped if .npy already exists)
    2. Unit tests
    3. Training (ISL)
    4. Evaluation (ISL)

Usage:
    cd signbridge
    python scripts/run_phase_2.py

    # Skip pose extraction if .npy already exists (default behaviour)
    # Force re-extraction:
    python scripts/run_phase_2.py --force-extract
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

SIGNBRIDGE_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_EXTRACTED = settings.processed_data_dir.parent / "INCLUDE"
SEQ_NPY = settings.processed_data_dir / "isl_sequences_include.npy"
LAB_NPY = settings.processed_data_dir / "isl_labels_include.npy"
CHECKPOINT = settings.model_dir / "best_isl.pt"
ISL_CONFIG = SIGNBRIDGE_ROOT / "backend" / "ml" / "training" / "configs" / "isl_config.yaml"


def run(cmd: list[str], step: str) -> None:
    """Run a subprocess command, exit on failure.

    Args:
        cmd: Command list to pass to subprocess.run.
        step: Human-readable step name for logging.
    """
    logger.info(f"{'='*60}")
    logger.info(f"STEP: {step}")
    logger.info(f"CMD:  {' '.join(cmd)}")
    logger.info(f"{'='*60}")
    result = subprocess.run(cmd, cwd=SIGNBRIDGE_ROOT)
    if result.returncode != 0:
        logger.error(f"FAILED: {step} (exit code {result.returncode})")
        sys.exit(result.returncode)
    logger.info(f"PASSED: {step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Phase 2 steps.")
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-run pose extraction even if .npy files already exist.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests (not recommended).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    # ── Step 1: Pose extraction ────────────────────────────────────────────────
    if args.force_extract or not (SEQ_NPY.exists() and LAB_NPY.exists()):
        if not INCLUDE_EXTRACTED.exists():
            logger.error(
                f"INCLUDE videos not found at {INCLUDE_EXTRACTED}.\n"
                "Move extracted category folders there first:\n"
                r'  Move-Item "signbridge\data\processed\_include_extracted\*" "Datasets\INCLUDE\"'
            )
            sys.exit(1)
        run(
            [python, "-c",
             f"from pathlib import Path; from backend.ml.data.loader import DatasetLoader; "
             f"loader = DatasetLoader(); loader.process_include_from_extracted(Path(r'{INCLUDE_EXTRACTED}')); loader.close()"],
            "Pose extraction — INCLUDE",
        )
    else:
        logger.info(f"Skipping pose extraction — .npy files already exist:\n  {SEQ_NPY}")

    # ── Step 2: Unit tests ─────────────────────────────────────────────────────
    if not args.skip_tests:
        run(
            [python, "-m", "pytest", "backend/tests/test_model.py", "-v", "--tb=short"],
            "Unit tests — Phase 2",
        )

    # ── Step 3: Training ───────────────────────────────────────────────────────
    run(
        [python, "scripts/train.py", "--language", "ISL", "--config", str(ISL_CONFIG)],
        "Training — ISL (CNN+BiLSTM)",
    )

    # ── Step 4: Evaluation ─────────────────────────────────────────────────────
    if not args.skip_eval:
        if not CHECKPOINT.exists():
            logger.error(f"Checkpoint not found: {CHECKPOINT}. Training may have failed.")
            sys.exit(1)
        run(
            [python, "scripts/evaluate.py", "--language", "ISL", "--checkpoint", str(CHECKPOINT)],
            "Evaluation — ISL",
        )

    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
