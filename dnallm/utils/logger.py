"""
DNALLM Logging Configuration

This module provides a centralized logging configuration for the DNALLM
project.
It replaces print statements with proper logging for better production
readiness.
"""

import logging
import sys
from pathlib import Path
import colorama
from colorama import Fore, Style
from typing import ClassVar

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)


class DNALLMLogger:
    """Centralized logger for DNALLM with colored output and structured
    logging."""

    def __init__(self, name: str = "dnallm", level: str = "INFO"):
        """
        Initialize the DNALLM logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Set up console and file handlers."""
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for detailed logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "dnallm.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)

    def success(self, message: str, **kwargs):
        """Log success message with green color."""
        self.logger.info(
            f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}", **kwargs
        )

    def failure(self, message: str, **kwargs):
        """Log failure message with red color."""
        self.logger.error(f"{Fore.RED}❌ {message}{Style.RESET_ALL}", **kwargs)

    def progress(self, message: str, **kwargs):
        """Log progress message with blue color."""
        self.logger.info(f"{Fore.BLUE}🔄 {message}{Style.RESET_ALL}", **kwargs)

    def warning_icon(self, message: str, **kwargs):
        """Log warning message with yellow color and icon."""
        self.logger.warning(
            f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}", **kwargs
        )

    def info_icon(self, message: str, **kwargs):
        """Log info message with cyan color and icon."""
        self.logger.info(f"{Fore.CYAN}i  {message}{Style.RESET_ALL}", **kwargs)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": Fore.WHITE,
        "INFO": Fore.CYAN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


# Global logger instance
_logger_instance: DNALLMLogger | None = None


def get_logger(name: str = "dnallm", level: str = "INFO") -> DNALLMLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        DNALLMLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DNALLMLogger(name, level)
    return _logger_instance


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """
    Set up logging configuration for the entire application.

    Args:
        level: Logging level
        log_file: Optional log file path
    """
    logger = get_logger(level=level)

    if log_file:
        # Add additional file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.logger.addHandler(file_handler)

    return logger


# Convenience functions for backward compatibility
def log_info(message: str):
    """Log info message."""
    get_logger().info(message)


def log_error(message: str):
    """Log error message."""
    get_logger().error(message)


def log_warning(message: str):
    """Log warning message."""
    get_logger().warning(message)


def log_debug(message: str):
    """Log debug message."""
    get_logger().debug(message)


def log_success(message: str):
    """Log success message."""
    get_logger().success(message)


def log_failure(message: str):
    """Log failure message."""
    get_logger().failure(message)


def log_progress(message: str):
    """Log progress message."""
    get_logger().progress(message)


# Context manager for temporary logging level changes
class LoggingContext:
    """Context manager for temporary logging level changes."""

    def __init__(self, level: str):
        self.level = level
        self.original_level = None

    def __enter__(self):
        self.original_level = get_logger().logger.level
        get_logger().logger.setLevel(getattr(logging, self.level.upper()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        get_logger().logger.setLevel(self.original_level)


# Decorator for logging function calls
def log_function_call(func):
    """Decorator to log function calls."""

    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(
            f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
        )
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise

    return wrapper
