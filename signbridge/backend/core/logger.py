"""SignBridge structured logger.

Provides a pre-configured loguru Logger instance for use across all
backend modules. Log level is read from the environment via settings.

Usage::

    from backend.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("KeypointExtractor initialised.")
"""

import sys
from functools import lru_cache

from loguru import logger as _loguru_logger

from backend.core.config import settings

# Remove the default loguru sink so we control formatting ourselves.
_loguru_logger.remove()

# ── Console sink ─────────────────────────────────────────────────────────────
_loguru_logger.add(
    sys.stderr,
    level=settings.log_level.upper(),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# ── File sink (rotates daily, kept for 7 days) ───────────────────────────────
_log_dir = settings.project_root / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

_loguru_logger.add(
    _log_dir / "signbridge_{time:YYYY-MM-DD}.log",
    level=settings.log_level.upper(),
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    ),
    rotation="00:00",  # rotate at midnight
    retention="7 days",
    compression="zip",
    backtrace=True,
    diagnose=False,  # no variable inspection in file logs
    enqueue=True,  # thread-safe async writes
)


@lru_cache(maxsize=None)
def get_logger(name: str) -> "loguru.Logger":  # type: ignore[name-defined]  # noqa: F821
    """Return a loguru logger bound to the given module name.

    Loguru does not have per-module loggers like the stdlib; this function
    binds the module name as extra context so log lines are traceable.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A loguru logger instance with the module name bound.
    """
    return _loguru_logger.bind(module=name)
