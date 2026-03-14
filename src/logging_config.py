"""
Logging configuration for SyntheticHealthSimulator.
Provides structured, configurable logging for production and development.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOGGER_NAME = "synthetic_health"


def setup_logging(
        level: int = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        format_string: Optional[str] = None,
        logger_name: str = DEFAULT_LOGGER_NAME,
) -> logging.Logger:
    """
    Configure logging for the synthetic health simulator.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for file logging. If provided, logs will be written
                  to this file in addition to console.
        format_string: Custom format string. If None, uses default format.
        logger_name: Name of the logger to configure. Defaults to 'synthetic_health'.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging(level=logging.DEBUG)
        >>> logger.info("Simulation started")
    """
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set format
    fmt = format_string or DEFAULT_LOG_FORMAT
    formatter = logging.Formatter(fmt)

    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # Log warning but don't crash
            logger.warning(f"Could not create file handler for {log_path}: {e}")

    logger.setLevel(level)
    logger.propagate = False  # Prevent propagation to root logger

    return logger


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """
    Get the configured logger instance.

    If the logger has no handlers, it will be configured with default settings
    (INFO level, console only).

    Args:
        name: Name of the logger. Defaults to 'synthetic_health'.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # No handlers – configure with defaults
        return setup_logging(logger_name=name)
    return logger


class ProgressLogger:
    """
    Utility for logging progress of long-running operations.

    Example:
        >>> progress = ProgressLogger(total_steps=5, operation="Generation")
        >>> progress.step(1, "Generating cohort")
        >>> progress.step(2, "Simulating lifestyle")
        >>> progress.finish()
    """

    def __init__(self, total_steps: int, operation: str = "Operation", logger: Optional[logging.Logger] = None):
        """
        Initialize progress logger.

        Args:
            total_steps: Total number of steps.
            operation: Name of the operation (used in log messages).
            logger: Logger instance to use. If None, uses default logger.
        """
        self.logger = logger or get_logger()
        self.total_steps = total_steps
        self.operation = operation
        self.current_step = 0

    def step(self, step_number: int, message: str) -> None:
        """
        Log progress for a specific step.

        Args:
            step_number: Current step number (1-based).
            message: Description of the step.
        """
        self.current_step = step_number
        percent = (step_number / self.total_steps) * 100
        self.logger.info(
            f"[{self.operation}] Step {step_number}/{self.total_steps} ({percent:.0f}%): {message}"
        )

    def finish(self, message: Optional[str] = None) -> None:
        """
        Log completion.

        Args:
            message: Optional custom completion message.
        """
        if message is None:
            message = f"{self.operation} completed successfully"
        self.logger.info(f"[{self.operation}] 100%: {message}")
