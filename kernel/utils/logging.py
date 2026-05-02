from __future__ import annotations

import logging
import sys
from typing import Optional


_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    force: bool = False,
) -> None:
    """Configure project-wide logging.

    Parameters
    ----------
    level:
        Logging level, e.g. logging.INFO or logging.DEBUG.
    fmt:
        Log message format string.
    datefmt:
        Datetime format string.
    force:
        If True, reconfigure existing root handlers.
    """
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stdout,
        force=force,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name if name is not None else "neural_kernel")


def set_log_level(level: int) -> None:
    """Set root logger level."""
    logging.getLogger().setLevel(level)


def add_file_handler(
    log_path: str,
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """Attach a file handler to the root logger."""
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root_logger.addHandler(file_handler)