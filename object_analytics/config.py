"""
Configuration Module.

Centralized configuration management for the Real-Time Object Analytics Engine.
All configurable parameters are defined here for easy maintenance and deployment.
"""

from typing import List, Tuple
import os
from .logging_config import logger

# =============================================================================
# DETECTION CONFIGURATION
# =============================================================================

# YOLO Model Configuration
MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
"""Path to the YOLO model file."""

CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
"""Detection confidence threshold (0.0-1.0)."""

# =============================================================================
# ANALYTICS CONFIGURATION
# =============================================================================

IMPORTANT_CLASSES: List[str] = [
    "person", "car", "truck", "bus", "motorbike", "bicycle"
]
"""Object classes to track and analyze for priority scoring."""

PRIORITY_WEIGHTS: Tuple[float, float] = (1000.0, 0.01)
"""Priority scoring weights: (area_weight, distance_penalty)."""

# =============================================================================
# VIDEO CONFIGURATION
# =============================================================================

CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
"""Default camera device index."""

FRAME_SIZE: Tuple[int, int] = (640, 640)
"""Target frame size for processing (width, height)."""

DISPLAY_WINDOW_NAME: str = "Real-Time Object Analytics"
"""Name of the display window."""

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

MAX_DISPLAY_DETECTIONS: int = 3
"""Maximum number of detections to display in console output."""

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

FPS_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green
"""Color for FPS display text (BGR)."""

LATENCY_COLOR: Tuple[int, int, int] = (255, 0, 0)  # Red
"""Color for latency display text (BGR)."""

TOP_DETECTION_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Red
"""Color for top priority detection bounding box (BGR)."""

OTHER_DETECTION_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green
"""Color for other detections bounding box (BGR)."""

TEXT_SCALE: float = 0.8
"""Font scale for overlay text."""

TEXT_THICKNESS: int = 2
"""Thickness for overlay text."""

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
"""Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""Logging format string."""

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config() -> None:
    """
    Validate configuration parameters.

    Checks all configuration values for validity and logs warnings or errors
    for invalid configurations.

    Raises
    ------
    ValueError
        If critical configuration parameters are invalid.
    """
    logger.debug("Validating configuration parameters...")

    # Validate confidence threshold
    if not (0.0 <= CONFIDENCE_THRESHOLD <= 1.0):
        error_msg = f"CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {CONFIDENCE_THRESHOLD}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate camera index
    if CAMERA_INDEX < 0:
        error_msg = f"CAMERA_INDEX must be non-negative, got {CAMERA_INDEX}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate frame size
    if FRAME_SIZE[0] <= 0 or FRAME_SIZE[1] <= 0:
        error_msg = f"FRAME_SIZE dimensions must be positive, got {FRAME_SIZE}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate max display detections
    if MAX_DISPLAY_DETECTIONS < 0:
        error_msg = f"MAX_DISPLAY_DETECTIONS must be non-negative, got {MAX_DISPLAY_DETECTIONS}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate priority weights
    if PRIORITY_WEIGHTS[0] <= 0 or PRIORITY_WEIGHTS[1] < 0:
        error_msg = f"PRIORITY_WEIGHTS must be positive, got {PRIORITY_WEIGHTS}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log configuration summary
    logger.info("Configuration validation successful")
    logger.debug(f"Model path: {MODEL_PATH}")
    logger.debug(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.debug(f"Important classes: {IMPORTANT_CLASSES}")
    logger.debug(f"Camera index: {CAMERA_INDEX}")
    logger.debug(f"Frame size: {FRAME_SIZE}")


def load_config_from_env() -> None:
    """
    Load configuration from environment variables with validation.

    Attempts to load configuration from environment variables and validates
    the loaded values. Logs warnings for invalid environment variables.
    """
    logger.debug("Loading configuration from environment variables...")

    # Environment variable loading with error handling
    try:
        if "YOLO_MODEL_PATH" in os.environ:
            global MODEL_PATH
            MODEL_PATH = os.environ["YOLO_MODEL_PATH"]
            logger.info(f"Loaded MODEL_PATH from environment: {MODEL_PATH}")

        if "CONFIDENCE_THRESHOLD" in os.environ:
            global CONFIDENCE_THRESHOLD
            CONFIDENCE_THRESHOLD = float(os.environ["CONFIDENCE_THRESHOLD"])
            logger.info(f"Loaded CONFIDENCE_THRESHOLD from environment: {CONFIDENCE_THRESHOLD}")

        if "CAMERA_INDEX" in os.environ:
            global CAMERA_INDEX
            CAMERA_INDEX = int(os.environ["CAMERA_INDEX"])
            logger.info(f"Loaded CAMERA_INDEX from environment: {CAMERA_INDEX}")

        if "LOG_LEVEL" in os.environ:
            global LOG_LEVEL
            LOG_LEVEL = os.environ["LOG_LEVEL"].upper()
            logger.info(f"Loaded LOG_LEVEL from environment: {LOG_LEVEL}")

    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to load configuration from environment: {e}")
        logger.warning("Using default configuration values")


# Load configuration from environment on import
load_config_from_env()

# Validate configuration on import
validate_config()