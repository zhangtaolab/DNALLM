"""Tests for the centralized logging module (dnallm.utils.logger).

Covers the DNALLMLogger constructor/handler setup, the get_logger singleton,
setup_logging's extra file handler, the backward-compatibility log functions,
the LoggingContext context manager, and the log_function_call decorator.

The logger is stdlib-logging based (not loguru): per-test hygiene means
removing handlers added to fresh loggers and keeping the singleton's handler
set unchanged across tests. Tests that may construct a fresh DNALLMLogger
chdir into tmp_path first because _setup_handlers creates logs/dnallm.log
under the CWD (FIX-04 artifact discipline).
"""

from __future__ import annotations

import logging

import pytest

from dnallm.utils import logger as logger_module
from dnallm.utils.logger import (
    DNALLMLogger,
    LoggingContext,
    get_logger,
    log_debug,
    log_error,
    log_failure,
    log_function_call,
    log_info,
    log_progress,
    log_success,
    log_warning,
    setup_logging,
)

_FRESH_LOGGER_NAMES: set[str] = set()


@pytest.fixture(autouse=True)
def _cleanup_fresh_loggers():
    """Close and remove handlers of every fresh logger created by a test."""
    yield
    for name in list(_FRESH_LOGGER_NAMES):
        std_logger = logging.getLogger(name)
        for handler in list(std_logger.handlers):
            handler.close()
            std_logger.removeHandler(handler)
    _FRESH_LOGGER_NAMES.clear()


@pytest.fixture
def fresh_dnallm_singleton(monkeypatch, tmp_path):
    """Rebuild the module singleton on a handler-free 'dnallm' stdlib logger.

    Yields the stdlib logger; on teardown every handler added during the test
    is closed/removed and the pre-existing handlers are restored, and the
    module-level singleton is restored by monkeypatch.
    """
    monkeypatch.chdir(tmp_path)
    std_logger = logging.getLogger("dnallm")
    saved_handlers = list(std_logger.handlers)
    for handler in saved_handlers:
        std_logger.removeHandler(handler)
    monkeypatch.setattr(logger_module, "_logger_instance", None)
    # Construct once here so tests (and caplog) never race a construction that
    # would reset the level on the stdlib logger mid-test.
    get_logger()
    yield std_logger
    for handler in list(std_logger.handlers):
        handler.close()
        std_logger.removeHandler(handler)
    for handler in saved_handlers:
        std_logger.addHandler(handler)


def _fresh_logger(name: str, level: str = "INFO") -> DNALLMLogger:
    """Construct a DNALLMLogger on a unique name registered for cleanup."""
    _FRESH_LOGGER_NAMES.add(name)
    return DNALLMLogger(name, level)


class TestGetLogger:
    """Singleton semantics of get_logger."""

    def test_get_logger_returns_singleton_instance(self, fresh_dnallm_singleton):
        """Repeated calls return the identical DNALLMLogger object."""
        first = get_logger()
        second = get_logger()
        assert first is second
        assert isinstance(first, DNALLMLogger)


class TestDNALLMLogger:
    """Constructor level/branching behavior."""

    def test_invalid_level_falls_back_to_info(self, tmp_path, monkeypatch):
        """An unknown level name must configure INFO, not raise."""
        monkeypatch.chdir(tmp_path)
        fresh = _fresh_logger("dnallm.test.invalid-level", "TOTALLY_BOGUS")
        assert fresh.logger.level == logging.INFO

    def test_valid_level_is_applied(self, tmp_path, monkeypatch):
        """A known level name is applied to the stdlib logger."""
        monkeypatch.chdir(tmp_path)
        fresh = _fresh_logger("dnallm.test.debug-level", "DEBUG")
        assert fresh.logger.level == logging.DEBUG

    def test_fresh_logger_installs_console_and_file_handlers(self, tmp_path, monkeypatch):
        """A brand-new logger gets one INFO console and one DEBUG file handler."""
        monkeypatch.chdir(tmp_path)
        fresh = _fresh_logger("dnallm.test.handlers")

        handlers = fresh.logger.handlers
        console = [
            h
            for h in handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(console) == 1
        assert console[0].level == logging.INFO
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG

        fresh.info("hello-file-sink")
        assert (tmp_path / "logs" / "dnallm.log").exists()

    def test_existing_handlers_are_not_duplicated(self, tmp_path, monkeypatch):
        """When the stdlib logger already has handlers, none are added."""
        monkeypatch.chdir(tmp_path)
        name = "dnallm.test.prehandled"
        _FRESH_LOGGER_NAMES.add(name)
        std_logger = logging.getLogger(name)
        preinstalled = logging.NullHandler()
        std_logger.addHandler(preinstalled)

        DNALLMLogger(name)

        assert std_logger.handlers == [preinstalled]

    def test_critical_emits_record(self, tmp_path, monkeypatch, caplog):
        """critical() emits at CRITICAL level."""
        monkeypatch.chdir(tmp_path)
        fresh = _fresh_logger("dnallm.test.critical")
        with caplog.at_level(logging.CRITICAL, logger="dnallm.test.critical"):
            fresh.critical("kaboom")
        assert "kaboom" in caplog.text

    def test_info_icon_prefixes_message(self, tmp_path, monkeypatch, caplog):
        """info_icon() emits an info record carrying the icon prefix."""
        monkeypatch.chdir(tmp_path)
        fresh = _fresh_logger("dnallm.test.info-icon")
        with caplog.at_level(logging.INFO, logger="dnallm.test.info-icon"):
            fresh.info_icon("attention")
        assert "i  attention" in caplog.text

    def test_warning_icon_emits_warning_record(self, tmp_path, monkeypatch, caplog):
        """warning_icon() emits a warning record with the icon prefix."""
        monkeypatch.chdir(tmp_path)
        fresh = _fresh_logger("dnallm.test.warning-icon")
        with caplog.at_level(logging.WARNING, logger="dnallm.test.warning-icon"):
            fresh.warning_icon("careful")
        assert "careful" in caplog.text
        assert any(r.levelno == logging.WARNING for r in caplog.records)


class TestSetupLogging:
    """setup_logging's optional extra file handler."""

    def test_setup_logging_adds_extra_file_handler(self, fresh_dnallm_singleton, tmp_path):
        """A log_file argument installs one additional DEBUG file handler that works."""
        std_logger = fresh_dnallm_singleton
        handlers_before = list(std_logger.handlers)
        extra_log = tmp_path / "extra.log"

        returned = setup_logging(level="DEBUG", log_file=str(extra_log))

        assert returned is get_logger()
        added = [h for h in std_logger.handlers if h not in handlers_before]
        assert len(added) == 1
        assert isinstance(added[0], logging.FileHandler)

        returned.info("setup-hello")
        assert extra_log.exists()
        assert "setup-hello" in extra_log.read_text()

    def test_setup_logging_without_log_file_adds_no_handler(self, fresh_dnallm_singleton):
        """Without log_file the configuration path returns the logger unchanged."""
        std_logger = fresh_dnallm_singleton
        handlers_before = list(std_logger.handlers)

        returned = setup_logging(level="INFO")

        assert returned is get_logger()
        assert list(std_logger.handlers) == handlers_before


class TestConvenienceFunctions:
    """Backward-compatibility module-level log functions."""

    def test_every_convenience_function_emits_its_message(self, fresh_dnallm_singleton, caplog):
        """log_info/error/warning/debug/success/failure/progress all emit."""
        with caplog.at_level(logging.DEBUG, logger="dnallm"):
            log_info("m-info")
            log_error("m-error")
            log_warning("m-warning")
            log_debug("m-debug")
            log_success("m-success")
            log_failure("m-failure")
            log_progress("m-progress")

        for message in (
            "m-info",
            "m-error",
            "m-warning",
            "m-debug",
            "m-success",
            "m-failure",
            "m-progress",
        ):
            assert message in caplog.text


class TestLoggingContext:
    """Temporary level-switching context manager."""

    def test_level_changed_inside_and_restored_after(self, fresh_dnallm_singleton):
        """The context raises the level inside and restores it on exit."""
        active = get_logger()
        original_level = active.logger.level

        with LoggingContext("WARNING"):
            assert active.logger.level == logging.WARNING
        assert active.logger.level == original_level

    def test_invalid_level_falls_back_to_info(self, fresh_dnallm_singleton):
        """An unknown context level configures INFO and still restores on exit."""
        active = get_logger()
        original_level = active.logger.level

        with LoggingContext("NOT_A_LEVEL"):
            assert active.logger.level == logging.INFO
        assert active.logger.level == original_level


class TestLogFunctionCall:
    """Decorator logging function entry/completion/failure."""

    def test_logs_call_and_completion_and_returns_value(self, fresh_dnallm_singleton, caplog):
        """Successful calls pass through the return value and log both debug lines."""

        @log_function_call
        def double(x):
            return 2 * x

        with caplog.at_level(logging.DEBUG, logger="dnallm"):
            assert double(21) == 42

        assert "Calling double" in caplog.text
        assert "double completed successfully" in caplog.text

    def test_logs_failure_and_reraises(self, fresh_dnallm_singleton, caplog):
        """Exceptions are logged with the error text and re-raised unchanged."""

        @log_function_call
        def raiser():
            raise ValueError("boom")

        with caplog.at_level(logging.DEBUG, logger="dnallm"):
            with pytest.raises(ValueError, match="boom"):
                raiser()
        assert "raiser failed with error: boom" in caplog.text
