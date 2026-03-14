"""
Logging configuration for SyntheticHealthSimulator.
Provides structured, configurable logging for production and development.
"""

import logging
import sys
from typing import Optional


def setup_logging(
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the synthetic health simulator.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for file logging
        format_string: Custom format string (uses default if None)

    Returns:
        Configured logger instance

    Example:
        '>>> logger = setup_logging(level=logging.DEBUG)'
        '>>> logger.info("Simulation started")'
    """

    logger = logging.getLogger("synthetic_health")

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Default format includes timestamp, level, and message
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.

    Returns existing logger or creates new with default settings.
    """
    logger = logging.getLogger("synthetic_health")
    if not logger.hasHandlers():
        return setup_logging()
    return logger


# Progress tracking utility
class ProgressLogger:
    """
    Utility for logging progress of long-running operations.

    Example:
        '>>> progress = ProgressLogger(total_steps=5, operation="Generation")'
        '>>> progress.step(1, "Generating cohort")'
        '>>> progress.step(2, "Simulating lifestyle")'
        '>>> progress.finish()'
    """

    def __init__(self, total_steps: int, operation: str = "Operation"):
        self.logger = get_logger()
        self.total_steps = total_steps
        self.operation = operation
        self.current_step = 0

    def step(self, step_number: int, message: str):
        """Log progress for a specific step."""
        self.current_step = step_number
        percent = (step_number / self.total_steps) * 100
        self.logger.info(
            f"[{self.operation}] Step {step_number}/{self.total_steps} "
            f"({percent:.0f}%): {message}"
        )

    def finish(self, message: Optional[str] = None):
        """Log completion."""
        if message is None:
            message = f"{self.operation} completed successfully"
        self.logger.info(f"[{self.operation}] 100%: {message}")
