################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice and this permission
# notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
Structured logging and error reporting for Tensile

Provides centralized logging configuration for all Tensile components,
with special handling for pytest-xdist workers.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


class WorkerAwareFilter(logging.Filter):
    """
    Adds worker ID to log records when running under pytest-xdist.

    This helps identify which worker process generated each log message.
    """

    def __init__(self):
        super().__init__()
        self.worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")

    def filter(self, record):
        record.worker_id = self.worker_id
        return True


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds color codes for different log levels.

    Colors are only applied when output is to a TTY.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, *args, use_color=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def format(self, record):
        if self.use_color and record.levelname in self.COLORS:
            # Add color to level name
            levelname_color = self.COLORS[record.levelname] + record.levelname + self.RESET
            record.levelname = levelname_color

        return super().format(record)


def setup_logging(
    name: str = "Tensile",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    worker_id: Optional[str] = None,
) -> logging.Logger:
    """
    Setup structured logging for Tensile components.

    This function configures both file and console logging with appropriate
    formatting and filtering. When running under pytest-xdist, each worker
    gets its own log file.

    Args:
        name: Logger name (default: "Tensile")
        level: Logging level (default: INFO)
        log_file: Path to log file (auto-generated if None)
        worker_id: Worker ID for pytest-xdist (auto-detected if None)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging("Tensile.Tests", logging.DEBUG)
        >>> logger.info("Test starting")
        >>> logger.error("Test failed", exc_info=True)
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Auto-detect worker ID if not provided
    if worker_id is None:
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")

    # Add worker filter
    worker_filter = WorkerAwareFilter()
    logger.addFilter(worker_filter)

    # Determine log file path
    if log_file is None:
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"tensile_{worker_id}.log"

    # File handler - always write detailed logs to file
    try:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)  # Always capture DEBUG in file

        # Detailed format for file
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - [%(worker_id)s] - [%(levelname)s] - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not create log file {log_file}: {e}", file=sys.stderr)

    # Console handler - only errors and critical to avoid cluttering test output
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.ERROR)

    # Concise format for console with color
    console_formatter = ColoredFormatter(
        fmt="[%(worker_id)s] %(levelname)s: %(message)s",
        use_color=True,
    )
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # Don't propagate to root logger to avoid duplicate messages
    logger.propagate = False

    logger.debug(f"Logging initialized: level={logging.getLevelName(level)}, file={log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    This is a convenience function that returns a child logger under the
    Tensile logger hierarchy. If the root Tensile logger hasn't been
    configured, it will be configured with default settings.

    Args:
        name: Module name (will be prefixed with "Tensile." if not already)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger("ClientWriter")
        >>> logger.info("Starting client execution")
    """
    if not name.startswith("Tensile."):
        name = f"Tensile.{name}"

    # Ensure root Tensile logger is configured
    root_logger = logging.getLogger("Tensile")
    if not root_logger.handlers:
        setup_logging("Tensile")

    return logging.getLogger(name)


def log_exception(logger: logging.Logger, message: str, exc_info=True):
    """
    Log an exception with full traceback and context.

    This is a helper function that ensures exceptions are logged with
    maximum detail for debugging.

    Args:
        logger: Logger instance to use
        message: Error message describing the context
        exc_info: Whether to include exception info (default: True)

    Example:
        >>> try:
        ...     dangerous_operation()
        ... except Exception:
        ...     log_exception(logger, "Operation failed")
    """
    logger.error(message, exc_info=exc_info, stack_info=True)


class LogContext:
    """
    Context manager for temporary log level changes.

    Useful for enabling debug logging for specific operations.

    Example:
        >>> logger = get_logger("Tests")
        >>> with LogContext(logger, logging.DEBUG):
        ...     # DEBUG messages will be shown in this block
        ...     logger.debug("Detailed information")
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.original_level = None

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        return False  # Don't suppress exceptions
