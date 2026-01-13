# Real-Time Object Analytics Engine for Live Video Streams

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

A Real-Time Object detection and analytics engine built with YOLOv8, designed for live video stream processing with performance monitoring, error handling, and structured output capabilities.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [API Reference](#api-reference)
- [Use Cases](#use-cases)
- [Testing](#testing)

## Features

- **Real-time Object Detection**: YOLOv8-powered detection with configurable confidence thresholds
- **Performance Monitoring**: FPS and latency measurement for production deployment
- **Semantic Filtering**: Focus on important object classes (person, car, truck, bus, motorbike, bicycle)
- **Priority Analytics**: Intelligent object prioritization based on size and position
- **Structured Output**: Integration-ready analytics summary for downstream systems
- **Robust Error Handling**: Graceful failure recovery and error logging
- **Configurable Architecture**: Easy parameter tuning for different deployment scenarios

## Architecture

```
Camera Frame
   ↓
Preprocessing (resize, color conversion)
   ↓
YOLO Inference (with latency measurement)
   ↓
Post-processing (NMS, confidence filtering)
   ↓
Semantic Filtering (important classes only)
   ↓
Priority Scoring (size + center proximity)
   ↓
Analytics Summary (structured output)
   ↓
Visualization + Real-time Display
```

### Core Components

- **`ObjectDetector`**: YOLOv8 wrapper with configurable confidence thresholds
- **`ObjectAnalytics`**: Priority scoring and structured analytics generation
- **`utils.py`**: Frame preprocessing and visualization utilities

## Requirements

- Python 3.12+
- Webcam or video input device
- ~2GB RAM for YOLOv8n model
- CPU/GPU with CUDA support (optional, for faster inference)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd real-time-object-analytics-engine-for-live-video-streams
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model (included):**
   - `yolov8n.pt` is already included in the repository

## Configuration

Edit the configuration constants in `object_analytics/main.py`:

```python
CONF_THRESHOLD = 0.4          # Detection confidence threshold (0.0-1.0)
IMPORTANT_CLASSES = [         # Classes to track
    "person", "car", "truck",
    "bus", "motorbike", "bicycle"
]
```

## Usage

### Basic Usage

Run the real-time analytics engine:

```bash
python -m object_analytics.main
```

### Expected Output

The system will display:
- **Live video feed** with bounding boxes and labels
- **FPS counter** (green) in top-left
- **Latency measurement** (blue) below FPS
- **Console analytics** showing top 3 detected objects
- **Structured summary** for integration

### Sample Console Output

```
Summary: {'total_objects': 3, 'top_object': {'class_id': 0, 'confidence': 0.85, 'priority': 15.2, 'center': (320, 240)}}
---- Analytics ----
Class: person, Conf: 0.85, AreaRatio: 0.1234, Priority: 15.20
Class: car, Conf: 0.72, AreaRatio: 0.0891, Priority: 12.45
Class: person, Conf: 0.68, AreaRatio: 0.0765, Priority: 10.12
```

## Performance Metrics

### Typical Performance (YOLOv8n on CPU)
- **FPS**: 15-30 fps
- **Latency**: 50-200ms per frame
- **Memory Usage**: ~1.5-2GB RAM
- **Accuracy**: Configurable (default: 40% confidence threshold)

### Performance Factors
- **Hardware**: GPU acceleration significantly improves FPS
- **Resolution**: Lower input resolution = higher FPS
- **Model Size**: Larger models (yolov8m, yolov8l) = higher accuracy, lower FPS
- **Confidence Threshold**: Higher threshold = fewer detections, higher FPS

## API Reference

### ObjectAnalytics.summarize(analytics_data)

Generates structured analytics summary for integration.

**Parameters:**
- `analytics_data` (list): List of analyzed detections

**Returns:**
- `dict`: Summary with total objects and top priority object details

**Example:**
```python
summary = analytics_engine.summarize(analyzed_detections)
# Returns: {'total_objects': 3, 'top_object': {...}}
```

## Use Cases

- **Security Systems**: Real-time monitoring and alerting
- **Traffic Analytics**: Vehicle counting and flow analysis
- **Retail Analytics**: Customer behavior tracking
- **Industrial Monitoring**: Equipment and personnel tracking
- **Drone Applications**: Aerial object detection and tracking

## Testing

The system includes built-in error handling and will gracefully handle:
- Camera disconnection
- Model loading failures
- Detection inference errors
- Memory constraints