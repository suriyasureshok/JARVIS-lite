"""
Real-Time Multimodal AI System - JARVIS-lite Main Module.

This module provides the main application entry point for the complete
multimodal AI system combining computer vision, analytics, LLM reasoning,
and voice interaction.

ARCHITECTURE:
Vision Pipeline: Camera → YOLO → Analytics → Display (Real-time, 30+ FPS)
Voice Pipeline: Microphone → STT → LLM → TTS → Speaker (Parallel, non-blocking)

The system processes video frames through a complete pipeline:
1. Frame capture and preprocessing
2. Object detection using YOLOv8
3. Semantic filtering for important classes
4. Analytics and priority scoring
5. LLM-powered reasoning via OpenRouter
6. Voice input/output interaction
7. Visualization with performance metrics

Examples
--------
Run the JARVIS-lite system:

>>> python -m object_analytics.main

Or import and use programmatically:

>>> from object_analytics.main import main
>>> main()  # Runs the multimodal system
"""

import cv2
import time
import signal
import sys
import threading
from contextlib import contextmanager
from typing import Optional, Generator

from .config import (
    CAMERA_INDEX, CONFIDENCE_THRESHOLD, IMPORTANT_CLASSES,
    DISPLAY_WINDOW_NAME, MAX_DISPLAY_DETECTIONS,
    FPS_COLOR, LATENCY_COLOR, TEXT_SCALE, TEXT_THICKNESS
)
from .logging_config import logger
from .utils import draw_detections, preprocess_frame, calculate_fps
from .detector import ObjectDetector
from .analytics import ObjectAnalytics
from .voice_engine import VoiceEngine
from .llm_engine import LLMEngine


class VideoAnalyticsApp:
    """
    Main application class for JARVIS-lite multimodal AI system.

    Encapsulates the complete multimodal pipeline:
    - Vision: Real-time object detection and analytics
    - Reasoning: LLM-powered intelligent question answering
    - Voice: Speech input/output for natural interaction
    
    Architecture:
    - Main thread: Vision processing loop (30+ FPS)
    - Voice thread: Speech I/O + LLM reasoning (non-blocking)
    """

    def __init__(self):
        """Initialize the JARVIS-lite application."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.detector: Optional[ObjectDetector] = None
        self.analytics_engine: Optional[ObjectAnalytics] = None
        self.llm_engine: Optional[LLMEngine] = None
        self.voice_engine: Optional[VoiceEngine] = None
        self.class_map: Optional[list] = None
        self.running = False
        self.latest_summary = {}  # Shared between vision and voice threads

    def initialize(self) -> bool:
        """
        Initialize all components of the analytics system.

        Returns
        -------
        bool
            True if initialization successful, False otherwise.
        """
        try:
            logger.info("Initializing Real-Time Object Analytics Engine...")

            # Initialize video capture
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera device {CAMERA_INDEX}")
                return False

            # Initialize detector
            self.detector = ObjectDetector(
                model_path="yolov8n.pt",
                conf=CONFIDENCE_THRESHOLD
            )
            # Convert class names dict to list for indexing by class_id
            class_names_dict = self.detector.model.names
            if class_names_dict:
                max_class_id = max(class_names_dict.keys())
                self.class_map = [class_names_dict.get(i, f'class_{i}') for i in range(max_class_id + 1)]
            else:
                self.class_map = []
            logger.info(f"Loaded {len(self.class_map)} class names from model")

            # Initialize analytics engine
            self.analytics_engine = ObjectAnalytics()

            # Initialize LLM reasoning engine (OpenRouter)
            try:
                self.llm_engine = LLMEngine()
                logger.info("LLM reasoning engine initialized successfully")
            except Exception as e:
                logger.warning(f"LLM engine initialization failed: {e}. Continuing with fallback reasoning.")
                self.llm_engine = None

            # Initialize voice engine
            vosk_model_path = "object_analytics/models/vosk-model-small-en-us-0.15"
            try:
                self.voice_engine = VoiceEngine(vosk_model_path)
                logger.info("Voice engine initialized successfully")
            except Exception as e:
                logger.warning(f"Voice engine initialization failed: {e}. Continuing without voice capabilities.")
                self.voice_engine = None

            logger.info("Initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.cleanup()
            return False

    def voice_loop(self):
        """
        Non-blocking voice interaction loop with LLM reasoning.

        Runs in a separate thread to handle:
        1. Speech-to-text (Vosk)
        2. LLM reasoning (OpenRouter)
        3. Text-to-speech (pyttsx3)
        
        This loop never blocks the main vision processing thread.
        """
        if not self.voice_engine:
            logger.warning("Voice engine not available, skipping voice loop")
            return
        
        if not self.llm_engine:
            logger.warning("LLM engine not available, voice interaction will be limited")

        logger.info("🎤 JARVIS voice interaction loop started")
        logger.info("📡 Using LLM-powered reasoning via OpenRouter")

        while self.running:
            try:
                # Listen for user input (blocking within voice thread only)
                user_question = self.voice_engine.listen()

                # Get latest scene summary (thread-safe copy)
                summary = self.latest_summary.copy()

                # Generate LLM response (with timeout protection)
                if self.llm_engine:
                    answer = self.llm_engine.answer(summary, user_question, timeout=5.0)
                else:
                    # Fallback to basic summary if LLM unavailable
                    total = summary.get('total_objects', 0)
                    answer = f"I see {total} object{'s' if total != 1 else ''} in the scene."

                # Output response
                logger.info(f"JARVIS: {answer}")
                self.voice_engine.speak(answer)

            except Exception as e:
                logger.error(f"Voice loop error: {e}")
                time.sleep(1)  # Brief pause before retrying

        logger.info("Voice interaction loop ended")

    def cleanup(self) -> None:
        """Clean up resources and close connections."""
        logger.info("Cleaning up resources...")

        if self.cap and self.cap.isOpened():
            self.cap.release()

        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

    @contextmanager
    def graceful_shutdown(self) -> Generator[None, None, None]:
        """Context manager for graceful shutdown handling."""
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            yield
        finally:
            self.cleanup()

    def process_frame(self, frame) -> tuple:
        """
        Process a single frame through the complete analytics pipeline.

        Parameters
        ----------
        frame : numpy.ndarray
            Input frame from video capture.

        Returns
        -------
        tuple
            (processed_frame, summary, fps, latency_ms)

        Raises
        ------
        ValueError
            If input frame is invalid.
        RuntimeError
            If processing fails.
        """
        if frame is None:
            error_msg = "Input frame cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not hasattr(frame, 'shape'):
            error_msg = f"Input frame must be a numpy array, got {type(frame)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            start_time = time.time()
            logger.debug(f"Processing frame with shape: {frame.shape}")

            # Preprocess frame
            processed_frame = preprocess_frame(frame)

            # Object detection with timing
            inference_start = time.time()
            try:
                detections = self.detector.detect_objects(processed_frame)
                inference_time = (time.time() - inference_start) * 1000
                logger.debug(f"Detection completed: {len(detections)} objects found in {inference_time:.1f}ms")
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                return frame, {"total_objects": 0, "top_object": None}, 0, 0

            # Filter important classes
            try:
                filtered_detections = [
                    det for det in detections
                    if self.class_map[det["class_id"]] in IMPORTANT_CLASSES
                ]
                logger.debug(f"Filtered to {len(filtered_detections)} important detections")
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Error filtering detections: {e}, using all detections")
                filtered_detections = detections

            # Analytics
            try:
                analyzed_detections = self.analytics_engine.analyze(
                    filtered_detections, frame.shape
                )
                summary = self.analytics_engine.summarize(analyzed_detections)
                logger.debug(f"Analytics completed: {summary['total_objects']} objects analyzed")
            except Exception as e:
                logger.error(f"Analytics failed: {e}")
                return frame, {"total_objects": 0, "top_object": None}, 0, 0

            # Calculate FPS
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0

            logger.debug(f"Frame processing completed in {processing_time:.3f}s ({fps:.1f} FPS)")
            return frame, summary, fps, inference_time

        except Exception as e:
            error_msg = f"Frame processing failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def display_results(self, frame, analyzed_detections, summary, fps, latency):
        """
        Display detection results and performance metrics on frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame to draw on.
        analyzed_detections : list
            Analyzed detections with priority scores.
        summary : dict
            Analytics summary.
        fps : float
            Current FPS.
        latency : float
            Inference latency in milliseconds.

        Raises
        ------
        ValueError
            If input parameters are invalid.
        RuntimeError
            If display operations fail.
        """
        if frame is None:
            error_msg = "Input frame cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if analyzed_detections is None:
            error_msg = "Analyzed detections cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(analyzed_detections, list):
            error_msg = f"Analyzed detections must be a list, got {type(analyzed_detections)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if summary is None:
            error_msg = "Summary cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(summary, dict):
            error_msg = f"Summary must be a dict, got {type(summary)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.debug(f"Displaying results: {len(analyzed_detections)} detections, FPS: {fps:.1f}")

            # Draw detections
            frame_with_detections = draw_detections(
                frame, analyzed_detections, self.class_map
            )

            # Add performance metrics
            try:
                cv2.putText(
                    frame_with_detections,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    TEXT_SCALE,
                    FPS_COLOR,
                    TEXT_THICKNESS
                )

                cv2.putText(
                    frame_with_detections,
                    f"Latency: {int(latency)}ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    TEXT_SCALE,
                    LATENCY_COLOR,
                    TEXT_THICKNESS
                )
            except cv2.error as e:
                logger.warning(f"Failed to add performance metrics text: {e}")

            # Display frame
            try:
                cv2.imshow(DISPLAY_WINDOW_NAME, frame_with_detections)
            except cv2.error as e:
                logger.error(f"Failed to display frame: {e}")
                raise RuntimeError(f"Display failed: {e}") from e

            logger.debug("Results displayed successfully")

        except Exception as e:
            error_msg = f"Display results failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def log_analytics(self, analyzed_detections, summary):
        """
        Log analytics information to console.

        Parameters
        ----------
        analyzed_detections : list
            Analyzed detections with priority scores.
        summary : dict
            Analytics summary.

        Raises
        ------
        ValueError
            If input parameters are invalid.
        RuntimeError
            If logging operations fail.
        """
        if analyzed_detections is None:
            error_msg = "Analyzed detections cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(analyzed_detections, list):
            error_msg = f"Analyzed detections must be a list, got {type(analyzed_detections)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if summary is None:
            error_msg = "Summary cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(summary, dict):
            error_msg = f"Summary must be a dict, got {type(summary)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.info(f"Analytics Summary: {summary}")

            if analyzed_detections:
                logger.info("Top Detections:")
                for i, det in enumerate(analyzed_detections[:MAX_DISPLAY_DETECTIONS]):
                    try:
                        if not isinstance(det, dict):
                            logger.warning(f"Detection {i} is not a dict, skipping")
                            continue

                        required_keys = ["class_id", "confidence", "area_ratio", "priority"]
                        missing_keys = [key for key in required_keys if key not in det]
                        if missing_keys:
                            logger.warning(f"Detection {i} missing keys {missing_keys}, skipping")
                            continue

                        class_id = det["class_id"]
                        if not isinstance(class_id, int) or class_id < 0 or class_id >= len(self.class_map):
                            logger.warning(f"Detection {i} has invalid class_id: {class_id}, skipping")
                            continue

                        class_name = self.class_map[class_id]
                        confidence = det["confidence"]
                        area_ratio = det["area_ratio"]
                        priority = det["priority"]

                        logger.info(
                            f"  {i+1}. {class_name}: "
                            f"Conf={confidence:.2f}, "
                            f"Area={area_ratio:.4f}, "
                            f"Priority={priority:.2f}"
                        )
                    except (KeyError, IndexError, TypeError) as e:
                        logger.warning(f"Error logging detection {i}: {e}")
                        continue

            logger.debug("Analytics logging completed")

        except Exception as e:
            error_msg = f"Analytics logging failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def run(self) -> int:
        """
        Run the main video analytics loop.

        Returns
        -------
        int
            Exit code (0 for success, 1 for error).

        Raises
        ------
        RuntimeError
            If the analytics loop fails to start or encounters critical errors.
        """
        try:
            if not self.initialize():
                logger.error("Failed to initialize application")
                return 1

            self.running = True
            prev_time = time.time()
            frame_count = 0

            logger.info("Starting video analytics loop. Press 'q' to quit.")

            # Start voice interaction thread
            if self.voice_engine:
                voice_thread = threading.Thread(target=self.voice_loop)
                voice_thread.daemon = True
                voice_thread.start()
                logger.info("Voice interaction thread started")
            else:
                logger.info("Voice capabilities not available - continuing with visual analytics only")

            with self.graceful_shutdown():
                while self.running:
                    try:
                        ret, frame = self.cap.read()
                        if not ret:
                            logger.warning("Failed to read frame from camera")
                            break

                        if frame is None or frame.size == 0:
                            logger.warning("Received empty frame from camera")
                            continue

                        frame_count += 1
                        logger.debug(f"Processing frame {frame_count}")

                        # Process frame (this includes detection and analytics)
                        processed_frame, summary, fps, latency = self.process_frame(frame)

                        # Update latest summary for voice interaction
                        self.latest_summary = summary

                        # Get analyzed detections for display and logging
                        # (We need to redo the filtering and analysis since process_frame returns summary only)
                        try:
                            processed_for_detection = preprocess_frame(frame)
                            detections = self.detector.detect_objects(processed_for_detection)

                            filtered_detections = [
                                det for det in detections
                                if self.class_map[det["class_id"]] in IMPORTANT_CLASSES
                            ]

                            analyzed_detections = self.analytics_engine.analyze(
                                filtered_detections, frame.shape
                            )
                        except Exception as e:
                            logger.warning(f"Failed to get analyzed detections for display: {e}")
                            analyzed_detections = []

                        # Log analytics
                        try:
                            self.log_analytics(analyzed_detections, summary)
                        except Exception as e:
                            logger.warning(f"Analytics logging failed: {e}")

                        # Display results
                        try:
                            self.display_results(processed_frame, analyzed_detections, summary, fps, latency)
                        except Exception as e:
                            logger.error(f"Display failed: {e}")
                            # Continue running even if display fails

                        # Check for quit command
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("Quit command received")
                            self.running = False

                    except KeyboardInterrupt:
                        logger.info("Keyboard interrupt received")
                        self.running = False
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}")
                        # Continue running for non-critical errors
                        continue

            logger.info(f"Video analytics loop ended after processing {frame_count} frames")
            return 0

        except Exception as e:
            error_msg = f"Critical error in run loop: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


def main() -> int:
    """
    Main entry point for the Real-Time Object Analytics Engine.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    app = VideoAnalyticsApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())