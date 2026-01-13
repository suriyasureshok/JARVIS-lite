"""
Logging Configuration Module.

Provides centralized logging configuration for the Real-Time Object Analytics Engine
using Loguru for enhanced logging capabilities.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Remove default logger to avoid conflicts
logger.remove()

# Configuration constants
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "object_analytics.log"
MAX_LOG_SIZE = "10 MB"
LOG_RETENTION = "30 days"

def setup_logging() -> None:
    """
    Configure Loguru logger with file and console output.

    Sets up logging to both console and rotating log files with proper formatting
    and log levels. Creates log directory if it doesn't exist.

    Raises
    ------
    OSError
        If log directory cannot be created or log file cannot be written.
    """
    try:
        # Create log directory if it doesn't exist
        LOG_DIR.mkdir(exist_ok=True)

        # Convert log level string to loguru format
        log_level = LOG_LEVEL.upper()

        # Add file logger with rotation and retention
        logger.add(
            LOG_FILE,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation=MAX_LOG_SIZE,
            retention=LOG_RETENTION,
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Thread-safe
            catch=True     # Catch and log exceptions in logging itself
        )

        # Add console logger with color
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}",
            colorize=True,
            backtrace=True,
            diagnose=True,
            catch=True
        )

        logger.info("Logging system initialized successfully")
        logger.info(f"Log file: {LOG_FILE.absolute()}")
        logger.info(f"Log level: {log_level}")

    except Exception as e:
        # Fallback to basic console logging if file logging fails
        print(f"Failed to setup file logging: {e}")
        print("Falling back to console logging only")

        logger.add(
            sys.stdout,
            level="INFO",
            format="{time:HH:mm:ss} | {level} | {message}",
            colorize=True
        )

        logger.warning("Using fallback console logging due to configuration error")


# Initialize logging on import
setup_logging()


def get_logger(name: str):
    """
    Get a logger instance with the specified name.

    Parameters
    ----------
    name : str
        Name for the logger (usually __name__).

    Returns
    -------
    loguru.Logger
        Configured logger instance.
    """
    return logger.bind(name=name)