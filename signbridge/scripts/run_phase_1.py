"""Phase 1 runner — validates Phase 1 setup and runs Phase 1 tests.

Phase 1 deliverables (already complete):
    - MediaPipe pose extraction pipeline
    - FrameBuffer sliding window
    - DatasetLoader for all 4 datasets
    - 28 unit tests passing

Usage:
    cd signbridge
    python scripts/run_phase_1.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import get_logger

logger = get_logger(__name__)

SIGNBRIDGE_ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    python = sys.executable

    run(
        [python, "-m", "pytest", "backend/tests/test_pose.py",
         "backend/tests/test_buffer.py", "-v", "--tb=short"],
        "Unit tests — Phase 1 (pose + buffer)",
    )

    logger.info("Phase 1 validation complete.")


if __name__ == "__main__":
    main()
