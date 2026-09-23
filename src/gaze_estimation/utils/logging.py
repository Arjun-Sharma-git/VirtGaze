"""Structured logging setup for the gaze estimation system."""
from __future__ import annotations

import logging
import sys
from typing import Optional

_ROOT_LOGGER_NAME = "gaze_estimation"


def setup_logging(
    level: str = "INFO",
    fmt: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure the root package logger.

    Call once at application startup::

        from gaze_estimation.utils.logging import setup_logging
        logger = setup_logging(level="DEBUG")

    Args:
        level:    Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        fmt:      Custom format string. Defaults to a structured format.
        log_file: Optional path to write logs to (in addition to stdout).

    Returns:
        The configured root logger.
    """
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the *gaze_estimation* namespace.

    Example::

        logger = get_logger("pipeline.camera")
        logger.info("Camera started at 60 FPS")
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
