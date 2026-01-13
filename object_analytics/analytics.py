"""
Object Analytics Module.

This module provides analytics and prioritization capabilities for detected objects,
including priority scoring based on object size and position, and structured
summaries for integration with downstream systems.
"""

from typing import List, Dict, Tuple, Any
import logging
from .config import PRIORITY_WEIGHTS
from .logging_config import logger


class ObjectAnalytics:
    """
    Analytics engine for processing and prioritizing detected objects.

    This class provides methods to analyze detected objects, assign priority scores
    based on size and position, and generate structured summaries for integration.

    The priority scoring algorithm combines:
    - Relative object size (area ratio)
    - Distance from frame center (as relevance proxy)

    Parameters
    ----------
    priority_weights : tuple of float, optional
        Weights for priority calculation (area_weight, distance_penalty).
        Default uses values from config.

    Examples
    --------
    >>> analytics = ObjectAnalytics()
    >>> prioritized = analytics.analyze(detections, (480, 640))
    >>> summary = analytics.summarize(prioritized)
    """

    def __init__(self, priority_weights: Tuple[float, float] = None):
        """
        Initialize the ObjectAnalytics engine.

        Parameters
        ----------
        priority_weights : tuple of float, optional
            Weights for priority calculation (area_weight, distance_penalty).
            If None, uses values from configuration.

        Raises
        ------
        ValueError
            If priority_weights format is invalid.
        """
        try:
            self.priority_weights = priority_weights or PRIORITY_WEIGHTS

            if not isinstance(self.priority_weights, (tuple, list)) or len(self.priority_weights) != 2:
                error_msg = f"priority_weights must be a tuple/list of 2 floats, got {self.priority_weights}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            self.area_weight, self.distance_penalty = self.priority_weights

            if not all(isinstance(w, (int, float)) for w in self.priority_weights):
                error_msg = f"priority_weights must contain numeric values, got {self.priority_weights}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"ObjectAnalytics initialized with weights: area={self.area_weight}, distance_penalty={self.distance_penalty}")

        except Exception as e:
            error_msg = f"Failed to initialize ObjectAnalytics: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def analyze(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """
        Analyze detected objects and assign priority scores.

        Processes raw detections to add analytics metadata including area ratios,
        center coordinates, and priority scores. Objects are sorted by priority
        in descending order.

        Parameters
        ----------
        detections : list of dict
            List of detection dictionaries from ObjectDetector.detect_objects().
            Each detection must contain 'bbox' key with [x1, y1, x2, y2] coordinates.
        frame_shape : tuple of int
            Shape of the input frame as (height, width[, channels]).

        Returns
        -------
        list of dict
            Enhanced detection dictionaries sorted by priority, each containing:
            - All original detection fields
            - 'area_ratio': float, Ratio of object area to frame area
            - 'center': tuple of int, Object center coordinates (x, y)
            - 'priority': float, Priority score (higher = more important)

        Raises
        ------
        ValueError
            If detections or frame_shape are invalid.
        RuntimeError
            If analysis processing fails.

        Notes
        -----
        Priority score calculation:
        priority = (area_ratio * area_weight) - (center_distance * distance_penalty)

        This favors larger objects closer to the frame center.

        Examples
        --------
        >>> detections = detector.detect_objects(frame)
        >>> analyzed = analytics.analyze(detections, frame.shape)
        >>> top_object = analyzed[0]  # Highest priority object
        """
        if detections is None:
            error_msg = "Detections list cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(detections, list):
            error_msg = f"Detections must be a list, got {type(detections)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if frame_shape is None:
            error_msg = "Frame shape cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(frame_shape, (tuple, list)) or len(frame_shape) < 2:
            error_msg = f"Frame shape must be a tuple/list of at least 2 integers, got {frame_shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            h, w = frame_shape[:2]  # Handle both (h,w) and (h,w,c) tuples

            if h <= 0 or w <= 0:
                error_msg = f"Frame dimensions must be positive, got height={h}, width={w}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug(f"Analyzing {len(detections)} detections for frame shape {frame_shape}")

            prioritized = []

            for i, detection in enumerate(detections):
                try:
                    if not isinstance(detection, dict):
                        logger.warning(f"Detection {i} is not a dict, skipping: {type(detection)}")
                        continue

                    if "bbox" not in detection:
                        logger.warning(f"Detection {i} missing 'bbox' key, skipping")
                        continue

                    bbox = detection["bbox"]
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        logger.warning(f"Detection {i} has invalid bbox format: {bbox}")
                        continue

                    x1, y1, x2, y2 = bbox

                    # Validate bounding box coordinates
                    if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                        logger.warning(f"Detection {i} has non-numeric bbox coordinates: {bbox}")
                        continue

                    if x1 >= x2 or y1 >= y2:
                        logger.warning(f"Detection {i} has invalid bbox coordinates (x1 >= x2 or y1 >= y2): {bbox}")
                        continue

                    # Calculate analytics
                    box_area = (x2 - x1) * (y2 - y1)
                    frame_area = h * w

                    if frame_area == 0:
                        logger.warning(f"Detection {i} has zero frame area, skipping")
                        continue

                    area_ratio = box_area / frame_area

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    center_distance = abs(center_x - w//2) + abs(center_y - h//2)
                    priority_score = area_ratio * self.area_weight - center_distance * self.distance_penalty

                    # Create enhanced detection dict
                    enhanced_detection = detection.copy()
                    enhanced_detection["area_ratio"] = round(area_ratio, 4)
                    enhanced_detection["center"] = (center_x, center_y)
                    enhanced_detection["priority"] = round(priority_score, 2)

                    prioritized.append(enhanced_detection)

                except Exception as e:
                    logger.warning(f"Error processing detection {i}: {e}")
                    continue

            # Sort by priority descending
            prioritized.sort(key=lambda x: x["priority"], reverse=True)

            logger.debug(f"Analysis completed: {len(prioritized)} objects prioritized")
            return prioritized

        except Exception as e:
            error_msg = f"Analysis processing failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def summarize(self, analytics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate structured analytics summary for integration.

        Creates a summary dictionary suitable for consumption by downstream
        systems, APIs, or databases. Includes total object count and details
        of the highest-priority object.

        Parameters
        ----------
        analytics_data : list of dict
            List of analyzed detections from ObjectAnalytics.analyze().

        Returns
        -------
        dict
            Summary dictionary with the following structure:
            {
                'total_objects': int, Total number of detected objects,
                'top_object': dict or None, Details of highest priority object
            }

            When top_object is present, it contains:
            - 'class_id': int, COCO class ID
            - 'confidence': float, Detection confidence
            - 'priority': float, Priority score
            - 'center': tuple of int, Object center coordinates

        Raises
        ------
        ValueError
            If analytics_data is invalid.
        RuntimeError
            If summary generation fails.

        Examples
        --------
        >>> analyzed = analytics.analyze(detections, frame.shape)
        >>> summary = analytics.summarize(analyzed)
        >>> print(f"Total objects: {summary['total_objects']}")
        >>> if summary['top_object']:
        ...     print(f"Top object class: {summary['top_object']['class_id']}")
        """
        if analytics_data is None:
            error_msg = "Analytics data cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(analytics_data, list):
            error_msg = f"Analytics data must be a list, got {type(analytics_data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.debug(f"Generating summary for {len(analytics_data)} analyzed objects")

            summary = {
                "total_objects": len(analytics_data),
                "top_object": None
            }

            if analytics_data:
                try:
                    top = analytics_data[0]  # Already sorted by priority

                    # Validate top object has required fields
                    required_fields = ["class_id", "confidence", "priority", "center"]
                    missing_fields = [field for field in required_fields if field not in top]

                    if missing_fields:
                        logger.warning(f"Top object missing required fields: {missing_fields}")
                        # Try to extract what we can
                        top_object = {}
                        for field in required_fields:
                            if field in top:
                                top_object[field] = top[field]
                        summary["top_object"] = top_object if top_object else None
                    else:
                        summary["top_object"] = {
                            "class_id": top["class_id"],
                            "confidence": top["confidence"],
                            "priority": top["priority"],
                            "center": top["center"]
                        }

                except (KeyError, IndexError) as e:
                    logger.warning(f"Error extracting top object data: {e}")
                    summary["top_object"] = None

            logger.debug(f"Summary generated: {summary['total_objects']} total objects")
            return summary

        except Exception as e:
            error_msg = f"Summary generation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
