"""
Object Detection Module.

This module provides the ObjectDetector class for performing real-time
object detection using YOLOv8 models with configurable parameters.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
from .logging_config import logger
from .config import MODEL_PATH, CONFIDENCE_THRESHOLD


class ObjectDetector:
    """
    YOLOv8-based object detector for real-time video analytics.

    This class wraps Ultralytics YOLOv8 models to provide object detection
    capabilities with configurable confidence thresholds and model selection.

    Parameters
    ----------
    model_path : str, optional
        Path to the YOLO model file. Default is "yolov8n.pt".
    conf : float, optional
        Confidence threshold for detections (0.0 to 1.0). Default is 0.4.

    Attributes
    ----------
    model : YOLO
        The loaded YOLOv8 model instance.
    conf : float
        Current confidence threshold.
    class_names : list of str
        List of class names from the model.

    Examples
    --------
    >>> detector = ObjectDetector(model_path="yolov8n.pt", conf=0.5)
    >>> detections = detector.detect_objects(frame)
    """

    def __init__(self, model_path: str = None, conf: float = None):
        """
        Initialize the ObjectDetector with a YOLO model.

        Parameters
        ----------
        model_path : str, optional
            Path to the YOLO model file. If None, uses MODEL_PATH from config.
        conf : float, optional
            Confidence threshold for detections (0.0 to 1.0). If None, uses CONFIDENCE_THRESHOLD from config.

        Raises
        ------
        FileNotFoundError
            If the model file is not found at the specified path.
        ValueError
            If confidence threshold is not in valid range.
        RuntimeError
            If model loading fails for any other reason.
        """
        self.model_path = model_path or MODEL_PATH
        self.conf = conf if conf is not None else CONFIDENCE_THRESHOLD
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = []

        logger.info(f"Initializing ObjectDetector with model: {self.model_path}")
        logger.debug(f"Confidence threshold: {self.conf}")

        # Validate confidence threshold
        if not (0.0 <= self.conf <= 1.0):
            error_msg = f"Confidence threshold must be between 0.0 and 1.0, got {self.conf}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Load the model
            logger.debug("Loading YOLO model...")
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            logger.info(f"Model loaded successfully. Classes: {len(self.class_names)}")

        except FileNotFoundError as e:
            error_msg = f"Model file not found: {self.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load YOLO model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform object detection on the given frame.

        Processes a single frame and returns detected objects with their
        bounding boxes, confidence scores, and class IDs.

        Parameters
        ----------
        frame : numpy.ndarray
            Input frame as a numpy array (H, W, C) in RGB format.

        Returns
        -------
        list of dict
            List of detection dictionaries, each containing:
            - 'class_id': int, COCO class ID of detected object
            - 'confidence': float, Detection confidence score (0.0-1.0)
            - 'bbox': list of int, Bounding box coordinates [x1, y1, x2, y2]

        Raises
        ------
        ValueError
            If input frame is invalid or None.
        RuntimeError
            If detection inference fails.

        Examples
        --------
        >>> detections = detector.detect_objects(frame)
        >>> for det in detections:
        ...     print(f"Class: {det['class_id']}, Conf: {det['confidence']:.2f}")
        """
        if frame is None:
            error_msg = "Input frame is None"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(frame, np.ndarray):
            error_msg = f"Input frame must be numpy array, got {type(frame)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if frame.size == 0:
            error_msg = "Input frame is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.debug(f"Running detection on frame shape: {frame.shape}")

            # Run inference
            results = self.model(frame, conf=self.conf, verbose=False)

            detections = []
            total_detections = 0

            for result in results:
                if result.boxes is None:
                    continue

                for r in result.boxes:
                    try:
                        cls_id = int(r.cls[0])
                        conf = float(r.conf[0])
                        x1, y1, x2, y2 = map(int, r.xyxy[0])

                        # Validate bounding box coordinates
                        if x1 >= x2 or y1 >= y2:
                            logger.warning(f"Invalid bounding box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                            continue

                        # Validate class ID
                        if cls_id < 0 or cls_id >= len(self.class_names):
                            logger.warning(f"Invalid class ID: {cls_id}")
                            continue

                        detection = {
                            "class_id": cls_id,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        }
                        detections.append(detection)
                        total_detections += 1

                    except (IndexError, ValueError, TypeError) as e:
                        logger.warning(f"Error processing detection result: {e}")
                        continue

            logger.debug(f"Detection completed: {total_detections} objects found")
            return detections

        except Exception as e:
            error_msg = f"Detection inference failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_class_name(self, class_id: int) -> str:
        """
        Get the class name for a given class ID.

        Parameters
        ----------
        class_id : int
            Class ID to look up.

        Returns
        -------
        str
            Class name corresponding to the ID.

        Raises
        ------
        IndexError
            If class_id is out of range.
        """
        try:
            return self.class_names[class_id]
        except IndexError as e:
            error_msg = f"Class ID {class_id} is out of range [0, {len(self.class_names) - 1}]"
            logger.error(error_msg)
            raise IndexError(error_msg) from e