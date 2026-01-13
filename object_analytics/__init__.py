"""
Real-Time Object Analytics Engine - JARVIS-lite.

A production-ready multimodal AI system for real-time object detection,
analytics, and LLM-powered reasoning using YOLOv8, OpenRouter, and voice I/O.

This package provides modular components for:
- Object detection with configurable confidence thresholds
- Real-time analytics with priority scoring
- Voice input/output capabilities (offline)
- LLM-powered intelligent scene reasoning via OpenRouter
- Performance monitoring and error handling
- Integration-ready structured output

Features
--------
- Real-time YOLOv8 object detection
- Intelligent object prioritization
- Offline voice recognition (Vosk)
- Offline text-to-speech (pyttsx3)
- LLM reasoning engine (OpenRouter free models)
- Multimodal human-computer interaction
- Performance metrics (FPS, latency)
- Structured analytics output
- Configurable processing pipeline
- Professional logging and error handling

Examples
--------
Basic usage for multimodal AI interaction:

>>> from object_analytics import VideoAnalyticsApp
>>> app = VideoAnalyticsApp()
>>> app.run()  # Full JARVIS-lite experience

Programmatic usage:

>>> from object_analytics import ObjectDetector, ObjectAnalytics, VoiceEngine, LLMEngine
>>> detector = ObjectDetector(conf=0.4)
>>> analytics = ObjectAnalytics()
>>> voice = VoiceEngine("models/vosk-model-small-en-us-0.15")
>>> llm = LLMEngine()  # Requires OPENROUTER_API_KEY environment variable

Command-line usage:

>>> python -m object_analytics.main
"""

from .detector import ObjectDetector
from .analytics import ObjectAnalytics
from .utils import draw_detections, preprocess_frame
from .voice_engine import VoiceEngine
from .llm_engine import LLMEngine
from .main import VideoAnalyticsApp
from .config import (
    CONFIDENCE_THRESHOLD, IMPORTANT_CLASSES, MODEL_PATH,
    CAMERA_INDEX, FRAME_SIZE, DISPLAY_WINDOW_NAME
)
from .logging_config import logger

__version__ = "0.2.0"
__author__ = "Suriya Sureshkumar"
__all__ = [
    "ObjectDetector",
    "ObjectAnalytics",
    "VoiceEngine",
    "LLMEngine",
    "VideoAnalyticsApp",
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