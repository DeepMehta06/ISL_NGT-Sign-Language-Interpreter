"""Phase 2 runner — executes all Phase 2 steps in order with persistent status logging.

Steps:
    1a. Pose extraction — INCLUDE (Datasets/INCLUDE/ → Datasets/processed/)
    1b. Pose extraction — ISL_Dataset (ISL.zip → Datasets/processed/)
    2.  Unit tests
    3.  Training (ISL — uses INCLUDE .npy; ISL_Dataset merged in Phase 3)
    4.  Evaluation (ISL)

Status is written to runs/phase_2_status.json after each step so a restart
can resume from where it left off.

Usage:
    cd signbridge
    python scripts/run_phase_2.py

    # Force re-run from the beginning:
    python scripts/run_phase_2.py --reset

    # Resume from where it stopped (default when status file exists):
    python scripts/run_phase_2.py
"""

from __future__ import annotations

import json
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ml.data.loader import DatasetLoader

logger = get_logger(__name__)

SIGNBRIDGE_ROOT = Path(__file__).resolve().parents[1]

# ── INCLUDE paths ──────────────────────────────────────────────────────────────
INCLUDE_DIR  = settings.include_dataset_dir
SEQ_NPY      = settings.processed_data_dir / "isl_sequences_include.npy"
LAB_NPY      = settings.processed_data_dir / "isl_labels_include.npy"

# ── ISL_Dataset paths ──────────────────────────────────────────────────────────
ISL_ZIP          = settings.isl_dataset_dir / "ISL.zip"
ISL_SEQ_NPY      = settings.processed_data_dir / "isl_sequences_isl_dataset.npy"
ISL_LAB_NPY      = settings.processed_data_dir / "isl_labels_isl_dataset.npy"
ISL_EXTRACT_TMP  = settings.processed_data_dir / "_isl_extracted"

# ── Shared ─────────────────────────────────────────────────────────────────────
CHECKPOINT = settings.model_dir / "best_isl.pt"
ISL_CONFIG = SIGNBRIDGE_ROOT / "backend" / "ml" / "training" / "configs" / "isl_config.yaml"
STATUS_FILE = settings.runs_dir / "phase_2_status.json"


def load_status() -> dict:
    """Load existing run status from disk, or return a fresh status dict.

    Returns:
        Dict with keys: steps_completed, started_at, last_updated.
    """
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            status = json.load(f)
        logger.info(f"Loaded existing run status: {STATUS_FILE}")
        logger.info(f"  Started at:    {status.get('started_at')}")
        logger.info(f"  Last updated:  {status.get('last_updated')}")
        logger.info(f"  Steps done:    {status.get('steps_completed')}")
        return status
    return {
        "started_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "steps_completed": [],
        "step_details": {},
    }


def save_status(status: dict, step: str, detail: str = "") -> None:
    """Mark a step as complete and persist status to disk.

    Args:
        status: Current status dict (mutated in place).
        step: Step name to mark complete.
        detail: Optional extra info saved alongside the step.
    """
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if step not in status["steps_completed"]:
        status["steps_completed"].append(step)
    status["step_details"][step] = {
        "completed_at": datetime.now().isoformat(),
        "detail": detail,
    }
    status["last_updated"] = datetime.now().isoformat()
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)
    logger.info(f"Status saved: {STATUS_FILE}")


def run(cmd: list[str], step: str, status: dict, detail: str = "") -> None:
    """Run a subprocess command, save status on success, exit on failure.

    Args:
        cmd: Command list to pass to subprocess.run.
        step: Step name for logging and status tracking.
        status: Current status dict.
        detail: Optional detail string saved to status on completion.
    """
    logger.info(f"{'='*60}")
    logger.info(f"STEP: {step}")
    logger.info(f"{'='*60}")
    result = subprocess.run(cmd, cwd=SIGNBRIDGE_ROOT)
    if result.returncode != 0:
        logger.error(f"FAILED: {step} (exit code {result.returncode})")
        logger.error("Status saved. Re-run 'python scripts/run_phase_2.py' to resume.")
        sys.exit(result.returncode)
    save_status(status, step, detail)
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing status file and restart all steps from scratch.",
    )
    return parser.parse_args()


def step_include_extraction(status: dict) -> None:
    """Run INCLUDE pose extraction in-process via DatasetLoader.

    Args:
        status: Current status dict — updated on completion.
    """
    if not INCLUDE_DIR.exists() or not any(INCLUDE_DIR.iterdir()):
        logger.error(f"INCLUDE videos not found at {INCLUDE_DIR}.")
        sys.exit(1)

    logger.info(f"{'='*60}")
    logger.info("STEP: Pose extraction — INCLUDE")
    logger.info(f"      Source : {INCLUDE_DIR}")
    logger.info(f"      Output : {settings.processed_data_dir}")
    logger.info(f"{'='*60}")

    with DatasetLoader() as loader:
        seqs, labs = loader.process_include_from_extracted(INCLUDE_DIR)

    detail = f"sequences={seqs.shape} labels={labs.shape}"
    save_status(status, "pose_extraction_include", detail)
    logger.info(f"PASSED: Pose extraction — INCLUDE | {detail}")


def step_isl_dataset_extraction(status: dict) -> None:
    """Extract ISL.zip and run pose extraction in-process via DatasetLoader.

    Cleans up the temporary extracted folder after saving .npy files.

    Args:
        status: Current status dict — updated on completion.
    """
    if not ISL_ZIP.exists():
        logger.error(f"ISL.zip not found at {ISL_ZIP}.")
        sys.exit(1)

    logger.info(f"{'='*60}")
    logger.info("STEP: Pose extraction — ISL_Dataset")
    logger.info(f"      Source : {ISL_ZIP}")
    logger.info(f"      Output : {settings.processed_data_dir}")
    logger.info(f"{'='*60}")

    with DatasetLoader() as loader:
        seqs, labs = loader.process_isl_dataset()

    if ISL_EXTRACT_TMP.exists():
        shutil.rmtree(ISL_EXTRACT_TMP)
        logger.info(f"Cleaned up temp folder: {ISL_EXTRACT_TMP}")

    detail = f"sequences={seqs.shape} labels={labs.shape}"
    save_status(status, "pose_extraction_isl_dataset", detail)
    logger.info(f"PASSED: Pose extraction — ISL_Dataset | {detail}")


def main() -> None:
    args = parse_args()
    python = sys.executable

    if args.reset and STATUS_FILE.exists():
        STATUS_FILE.unlink()
        logger.info("Status file deleted. Starting fresh.")

    status = load_status()
    done = set(status["steps_completed"])

    # ── Step 1a: INCLUDE pose extraction ──────────────────────────────────────
    if "pose_extraction_include" not in done and (
        args.force_extract or not (SEQ_NPY.exists() and LAB_NPY.exists())
    ):
        step_include_extraction(status)
    else:
        logger.info("Skipping INCLUDE extraction — already complete or .npy exists.")

    # ── Step 1b: ISL_Dataset pose extraction ──────────────────────────────────
    if "pose_extraction_isl_dataset" not in done and (
        args.force_extract or not (ISL_SEQ_NPY.exists() and ISL_LAB_NPY.exists())
    ):
        step_isl_dataset_extraction(status)
    else:
        logger.info("Skipping ISL_Dataset extraction — already complete or .npy exists.")

    # ── Step 2: Unit tests ────────────────────────────────────────────────────
    if not args.skip_tests and "unit_tests" not in done:
        run(
            [python, "-m", "pytest", "backend/tests/test_model.py", "-v", "--tb=short"],
            "unit_tests",
            status,
        )
    elif "unit_tests" in done:
        logger.info("Skipping unit tests — already passed.")

    # ── Step 3: Training ──────────────────────────────────────────────────────
    if "training" not in done:
        run(
            [python, "scripts/train.py", "--language", "ISL", "--config", str(ISL_CONFIG)],
            "training",
            status,
            detail="ISL CNN+BiLSTM on INCLUDE",
        )
    else:
        logger.info("Skipping training — already complete.")

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    if not args.skip_eval and "evaluation" not in done:
        if not CHECKPOINT.exists():
            logger.error(f"Checkpoint not found: {CHECKPOINT}. Training may have failed.")
            sys.exit(1)
        run(
            [python, "scripts/evaluate.py", "--language", "ISL", "--checkpoint", str(CHECKPOINT)],
            "evaluation",
            status,
        )
    elif "evaluation" in done:
        logger.info("Skipping evaluation — already complete.")

    logger.info("=" * 60)
    logger.info("Phase 2 complete.")
    logger.info(f"Status file: {STATUS_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
