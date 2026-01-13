"""
Real-Time Object Analytics Engine.

A production-ready computer vision package for real-time object detection
and analytics on live video streams using YOLOv8.

This package provides modular components for:
- Object detection with configurable confidence thresholds
- Real-time analytics with priority scoring
- Performance monitoring and error handling
- Integration-ready structured output

Features
--------
- Real-time YOLOv8 object detection
- Intelligent object prioritization
- Performance metrics (FPS, latency)
- Structured analytics output
- Configurable processing pipeline
- Professional logging and error handling

Examples
--------
Basic usage for real-time object detection:

>>> from object_analytics import ObjectDetector, ObjectAnalytics
>>> detector = ObjectDetector(conf=0.4)
>>> analytics = ObjectAnalytics()
>>> # Process frames in real-time loop

Command-line usage:

>>> python -m object_analytics.main
"""

from .detector import ObjectDetector
from .analytics import ObjectAnalytics
from .utils import draw_detections, preprocess_frame
from .config import (
    CONFIDENCE_THRESHOLD, IMPORTANT_CLASSES, MODEL_PATH,
    CAMERA_INDEX, FRAME_SIZE, DISPLAY_WINDOW_NAME
)
from .logging_config import logger

__version__ = "0.1.0"
__author__ = "Suriya Sureshkumar"
__all__ = [
    "ObjectDetector",
    "ObjectAnalytics",
    "draw_detections",
    "preprocess_frame",
    "CONFIDENCE_THRESHOLD",
    "IMPORTANT_CLASSES",
    "MODEL_PATH",
    "CAMERA_INDEX",
    "FRAME_SIZE",
    "DISPLAY_WINDOW_NAME",
    "logger"
]