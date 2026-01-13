"""
Utility Functions Module.

This module provides utility functions for frame preprocessing and visualization
of object detection results in real-time video analytics.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional
import logging

from .config import (
    FRAME_SIZE, TOP_DETECTION_COLOR, OTHER_DETECTION_COLOR,
    TEXT_SCALE, TEXT_THICKNESS
)
from .logging_config import logger


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]],
                   class_names: List[str]) -> np.ndarray:
    """
    Draw bounding boxes and labels on the frame for each detection.

    Annotates the input frame with bounding boxes, class labels, confidence scores,
    and priority information for all detected objects. The top-priority object
    is highlighted in a distinct color.

    Parameters
    ----------
    frame : numpy.ndarray
        Input frame as BGR numpy array (H, W, 3) to draw on.
    detections : list of dict
        List of detection dictionaries from ObjectAnalytics.analyze().
        Each detection must contain 'bbox', 'class_id', 'confidence', and 'priority'.
    class_names : list of str
        List of class names indexed by class_id (e.g., detector.model.names).

    Returns
    -------
    numpy.ndarray
        Annotated frame with bounding boxes and labels drawn on it.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    RuntimeError
        If drawing operations fail.

    Notes
    -----
    The function modifies the input frame in-place and returns it.
    Top-priority object (index 0) is drawn in red, others in green.

    Examples
    --------
    >>> annotated_frame = draw_detections(frame, detections, class_names)
    >>> cv2.imshow('Detections', annotated_frame)
    """
    if frame is None:
        error_msg = "Input frame cannot be None"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(frame, np.ndarray):
        error_msg = f"Frame must be numpy array, got {type(frame)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if frame.size == 0:
        error_msg = "Input frame is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if detections is None:
        error_msg = "Detections list cannot be None"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(detections, list):
        error_msg = f"Detections must be a list, got {type(detections)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if class_names is None:
        error_msg = "Class names list cannot be None"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(class_names, list):
        error_msg = f"Class names must be a list, got {type(class_names)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.debug(f"Drawing {len(detections)} detections on frame shape {frame.shape}")

        for i, detection in enumerate(detections):
            try:
                if not isinstance(detection, dict):
                    logger.warning(f"Detection {i} is not a dict, skipping: {type(detection)}")
                    continue

                # Validate required keys
                required_keys = ["bbox", "class_id", "confidence", "priority"]
                missing_keys = [key for key in required_keys if key not in detection]
                if missing_keys:
                    logger.warning(f"Detection {i} missing required keys: {missing_keys}, skipping")
                    continue

                bbox = detection["bbox"]
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    logger.warning(f"Detection {i} has invalid bbox format: {bbox}, skipping")
                    continue

                x1, y1, x2, y2 = bbox

                # Validate bounding box coordinates
                if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                    logger.warning(f"Detection {i} has non-numeric bbox coordinates: {bbox}, skipping")
                    continue

                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"Detection {i} has invalid bbox coordinates (x1 >= x2 or y1 >= y2): {bbox}, skipping")
                    continue

                # Validate class_id
                class_id = detection["class_id"]
                if not isinstance(class_id, int) or class_id < 0 or class_id >= len(class_names):
                    logger.warning(f"Detection {i} has invalid class_id: {class_id}, skipping")
                    continue

                confidence = detection["confidence"]
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    logger.warning(f"Detection {i} has invalid confidence: {confidence}, skipping")
                    continue

                # Choose color based on priority rank
                color = TOP_DETECTION_COLOR if i == 0 else OTHER_DETECTION_COLOR

                # Create label text
                priority = detection.get("priority", "N/A")
                label = f"{class_names[class_id]} {confidence:.2f} P:{priority}"

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw label background for better readability
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS
                )

                # Ensure label doesn't go outside frame bounds
                label_x1 = max(0, int(x1))
                label_y1 = max(label_height + 5, int(y1) - 5)
                label_x2 = min(frame.shape[1], label_x1 + label_width)
                label_y2 = min(frame.shape[0], label_y1 + label_height + 5)

                cv2.rectangle(frame, (label_x1, label_y1 - label_height - 5),
                             (label_x2, label_y1), color, -1)

                # Draw label text
                cv2.putText(frame, label, (label_x1, label_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)

            except Exception as e:
                logger.warning(f"Error drawing detection {i}: {e}")
                continue

        logger.debug("Detection drawing completed successfully")
        return frame

    except Exception as e:
        error_msg = f"Drawing detections failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def preprocess_frame(frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Preprocess the frame for object detection.

    Resizes the input frame to the target size and converts color space
    from BGR to RGB as required by YOLO models.

    Parameters
    ----------
    frame : numpy.ndarray
        Input frame as BGR numpy array (H, W, 3).
    target_size : tuple of int, optional
        Target size for resizing as (width, height). Default uses FRAME_SIZE from config.

    Returns
    -------
    numpy.ndarray
        Preprocessed frame as RGB numpy array with target_size dimensions.

    Raises
    ------
    ValueError
        If input frame or target_size are invalid.
    RuntimeError
        If preprocessing operations fail.

    Notes
    -----
    YOLO models expect RGB input, so BGR to RGB conversion is performed.
    The function uses OpenCV's default interpolation for resizing.

    Examples
    --------
    >>> processed = preprocess_frame(frame, target_size=(640, 640))
    >>> detections = detector.detect_objects(processed)
    """
    if frame is None:
        error_msg = "Input frame cannot be None"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(frame, np.ndarray):
        error_msg = f"Frame must be numpy array, got {type(frame)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if frame.size == 0:
        error_msg = "Input frame is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if frame.ndim != 3 or frame.shape[2] != 3:
        error_msg = f"Frame must be 3-channel BGR image, got shape {frame.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if target_size is None:
        target_size = FRAME_SIZE

    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
        error_msg = f"Target size must be a tuple/list of 2 integers, got {target_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    width, height = target_size
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        error_msg = f"Target size dimensions must be positive integers, got {target_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        logger.debug(f"Preprocessing frame from {frame.shape} to {target_size}")

        # Resize frame
        frame_resized = cv2.resize(frame, target_size)

        # Convert BGR to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        logger.debug("Frame preprocessing completed successfully")
        return frame_rgb

    except cv2.error as e:
        error_msg = f"OpenCV preprocessing failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Frame preprocessing failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def calculate_fps(prev_time: float) -> Tuple[float, float]:
    """
    Calculate frames per second and return current time.

    Parameters
    ----------
    prev_time : float
        Previous frame timestamp from time.time().

    Returns
    -------
    tuple of float
        (fps, current_time) where fps is frames per second and
        current_time is the current timestamp.

    Raises
    ------
    ValueError
        If prev_time is invalid.
    RuntimeError
        If time calculation fails.

    Examples
    --------
    >>> prev_time = time.time()
    >>> fps, curr_time = calculate_fps(prev_time)
    >>> print(f"FPS: {fps:.1f}")
    """
    if prev_time is None:
        error_msg = "Previous time cannot be None"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(prev_time, (int, float)):
        error_msg = f"Previous time must be a number, got {type(prev_time)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        curr_time = time.time()

        if prev_time <= 0:
            logger.warning(f"Invalid previous time: {prev_time}, returning 0 FPS")
            fps = 0.0
        else:
            time_diff = curr_time - prev_time
            if time_diff <= 0:
                logger.warning(f"Time difference is non-positive: {time_diff}, returning 0 FPS")
                fps = 0.0
            else:
                fps = 1.0 / time_diff

        logger.debug(f"FPS calculated: {fps:.2f}")
        return fps, curr_time

    except Exception as e:
        error_msg = f"FPS calculation failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e