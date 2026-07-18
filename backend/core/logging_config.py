"""
BridgeGuardian AI — Logging Configuration
Structured, coloured logging for development and production.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class ColorFormatter(logging.Formatter):
    """ANSI colour formatter for console output."""

    GREY = "\x1b[38;21m"
    BLUE = "\x1b[34m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        colour = self.FORMATS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            f"{colour}%(asctime)s{self.RESET} | "
            f"{colour}%(levelname)-8s{self.RESET} | "
            f"%(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a named logger.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        name: Logger name. Uses root logger if None.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name or "bridgeguardian")

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console Handler (Coloured)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # File Handler (Rotating, Plain Text)
    try:
        from backend.core.config import get_settings
        settings = get_settings()
        logs_dir = Path(settings.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_path = logs_dir / "app.log"
        
        file_handler = RotatingFileHandler(
            file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to set up file logging: {e}\n")

    logger.propagate = False

    return logger


# Module-level convenience logger
logger = setup_logging()

