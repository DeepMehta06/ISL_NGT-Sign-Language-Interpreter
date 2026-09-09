"""Master runner — executes ALL phases in sequence.

Imports and runs each phase's runner in order. Stops immediately if any
phase fails.

Usage:
    cd signbridge
    python scripts/run_all_phases.py

    # Start from a specific phase:
    python scripts/run_all_phases.py --from-phase 2

    # Run only one phase:
    python scripts/run_all_phases.py --only-phase 2
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import get_logger

logger = get_logger(__name__)

PHASES: dict[int, str] = {
    1: "scripts.run_phase_1",
    2: "scripts.run_phase_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all SignBridge phases in sequence.")
    parser.add_argument(
        "--from-phase",
        type=int,
        default=1,
        help="Start execution from this phase number (default: 1).",
    )
    parser.add_argument(
        "--only-phase",
        type=int,
        default=None,
        help="Run only this single phase number.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    phases_to_run = (
        {args.only_phase: PHASES[args.only_phase]}
        if args.only_phase
        else {k: v for k, v in PHASES.items() if k >= args.from_phase}
    )

    for phase_num, module_path in sorted(phases_to_run.items()):
        logger.info(f"{'#'*60}")
        logger.info(f"# PHASE {phase_num}")
        logger.info(f"{'#'*60}")
        try:
            module = importlib.import_module(module_path)
            module.main()
        except SystemExit as exc:
            if exc.code != 0:
                logger.error(f"Phase {phase_num} failed. Stopping.")
                sys.exit(exc.code)
        except Exception as exc:
            logger.error(f"Phase {phase_num} raised an exception: {exc}")
            sys.exit(1)

        logger.info(f"Phase {phase_num} complete.")

    logger.info("All phases complete.")


if __name__ == "__main__":
    main()
