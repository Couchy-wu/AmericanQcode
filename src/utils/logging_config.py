"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger

from src.core.config import load_config


def setup_logging() -> None:
    """Configure structured logging with loguru.

    - Console output for development
    - File output with rotation
    """
    cfg = load_config()
    log_config = cfg.get("logging", {})
    level = log_config.get("level", "INFO")
    log_file = log_config.get("file", "data/logs/app.log")
    rotation = log_config.get("rotation", "10 MB")

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention="30 days",
        compression="gz",
    )

    logger.info("Logging configured: level={}, file={}", level, log_file)


def get_logger(name: str):
    """Get a logger bound to a specific module name."""
    return logger.bind(name=name)
