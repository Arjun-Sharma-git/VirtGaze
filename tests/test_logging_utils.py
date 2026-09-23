"""Tests for logging setup and the namespaced logger factory.

``setup_logging`` mutates global logging state, so the fixture snapshots the
package logger's handlers and level and restores them afterwards.
"""

from __future__ import annotations

import logging

import pytest

from gaze_estimation.utils.logging import get_logger, setup_logging

ROOT_NAME = "gaze_estimation"


@pytest.fixture
def package_logger():
    """The package logger, with a clean slate that is restored on teardown."""
    logger = logging.getLogger(ROOT_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    logger.handlers.clear()

    yield logger

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.handlers.extend(saved_handlers)
    logger.setLevel(saved_level)


def test_returns_the_package_logger(package_logger):
    assert setup_logging() is package_logger


def test_sets_the_requested_level(package_logger):
    setup_logging(level="DEBUG")

    assert package_logger.level == logging.DEBUG


def test_unknown_level_falls_back_to_info(package_logger):
    setup_logging(level="NOT-A-LEVEL")

    assert package_logger.level == logging.INFO


def test_installs_a_single_console_handler(package_logger):
    setup_logging(level="INFO")

    assert len(package_logger.handlers) == 1
    assert isinstance(package_logger.handlers[0], logging.StreamHandler)


def test_repeated_calls_do_not_add_handlers(package_logger):
    setup_logging(level="INFO")
    first = list(package_logger.handlers)

    setup_logging(level="DEBUG")

    assert package_logger.handlers == first


def test_log_file_handler_writes_messages(package_logger, tmp_path):
    log_file = tmp_path / "gaze.log"

    setup_logging(level="INFO", log_file=str(log_file))
    get_logger("tests.logging").info("hello from the test")

    for handler in package_logger.handlers:
        handler.flush()
    assert "hello from the test" in log_file.read_text(encoding="utf-8")


def test_custom_format_is_used(package_logger, tmp_path):
    log_file = tmp_path / "custom.log"

    setup_logging(level="INFO", fmt="[%(levelname)s] %(message)s", log_file=str(log_file))
    get_logger("tests.logging").warning("formatted")

    for handler in package_logger.handlers:
        handler.flush()
    assert "[WARNING] formatted" in log_file.read_text(encoding="utf-8")


def test_get_logger_is_namespaced_under_the_package():
    logger = get_logger("pipeline.camera")

    assert logger.name == f"{ROOT_NAME}.pipeline.camera"
    assert logger is get_logger("pipeline.camera")  # logging caches by name
