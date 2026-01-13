# JARVIS-lite: LLM-Powered Multimodal AI System

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Vosk](https://img.shields.io/badge/Vosk-Offline-red.svg)](https://alphacephei.com/vosk/)
[![pyttsx3](https://img.shields.io/badge/pyttsx3-TTS-yellow.svg)](https://pyttsx3.readthedocs.io/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM-purple.svg)](https://openrouter.ai/)

A production-ready multimodal AI system combining real-time computer vision, intelligent analytics, LLM-powered reasoning, and voice interaction. Built with YOLOv8, OpenRouter free models, Vosk, and pyttsx3 for serious human-computer interaction.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Voice Commands](#voice-commands)
- [Performance Metrics](#performance-metrics)
- [API Reference](#api-reference)
- [Multimodal Architecture](#multimodal-architecture)
- [Troubleshooting](#troubleshooting)
- [Use Cases](#use-cases)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Real-time Object Detection**: YOLOv8-powered detection with configurable confidence thresholds
- **Intelligent Analytics**: Priority scoring based on object size and position
- **LLM Reasoning**: OpenRouter API with free models (Llama-3.3-70b, Mistral-7b)
- **Voice Input**: Offline speech recognition using Vosk
- **Voice Output**: Text-to-speech synthesis with pyttsx3
- **Conversational AI**: Natural language scene understanding and question answering
- **Multimodal I/O**: Vision + Voice + LLM integration
- **Non-blocking Architecture**: Voice and LLM run parallel to vision processing
- **Performance Monitoring**: FPS and latency measurement
- **Robust Error Handling**: Graceful failure recovery and fallback modes
- **Production-Ready**: Modular design with proper separation of concerns

## Architecture

```
Camera → YOLO (Perception) → Analytics (Meaning) → LLM (Reasoning) → Answer
   ↓                                ↓                      ↓           ↓
Voice Input ← STT (Vosk) ← User ← TTS (pyttsx3) ← Response ← OpenRouter
```

### Multimodal Pipeline

1. **Vision Pipeline** (Real-time Loop - Main Thread):
   - Camera capture → Preprocessing → YOLO inference → Post-processing → Analytics → Display
   - Target: 30+ FPS

2. **Voice Pipeline** (Parallel Thread - Non-blocking):
   - Microphone → Speech-to-Text (Vosk) → Question
   - Scene Summary → LLM (OpenRouter) → Answer
   - Answer → Text-to-Speech (pyttsx3) → Speaker

3. **Integration**:
   - Vision continuously updates shared scene summary
   - Voice queries latest scene state without blocking vision
   - LLM provides intelligent, grounded responses

### Core Components

- **`ObjectDetector`**: YOLOv8 wrapper with configurable confidence thresholds
- **`ObjectAnalytics`**: Priority scoring and structured analytics generation (LLM-optimized)
- **`LLMEngine`**: OpenRouter client with prompt engineering for grounded reasoning
- **`VoiceEngine`**: Offline speech recognition and synthesis
- **`VideoAnalyticsApp`**: Main application orchestrating all components with threading

## Requirements

- Python 3.12+
- Webcam or video input device
- Microphone for voice input
- Speakers/headphones for voice output
- ~2GB RAM for YOLOv8n model
- CPU/GPU with CUDA support (optional, for faster inference)
- **OpenRouter API key** (free - see Configuration)

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

4. **Download Vosk speech recognition model:**
   - Visit: [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
   - Download: `vosk-model-small-en-us-0.15.zip`
   - Extract to: `object_analytics/models/vosk-model-small-en-us-0.15/`

   ```bash
   # After downloading and extracting
   ls object_analytics/models/
   # Should show: vosk-model-small-en-us-0.15/
   ```

5. **Get OpenRouter API Key (FREE):**
   - Visit: [https://openrouter.ai/keys](https://openrouter.ai/keys)
   - Sign up for a free account
   - Generate an API key
   - Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` and add your API key:
   ```
   OPENROUTER_API_KEY=your_actual_key_here
   ```

## Configuration

### Core Settings

Edit `object_analytics/config.py` to customize:

```python
# Camera settings
CAMERA_INDEX = 0  # Default webcam

# Detection parameters
CONFIDENCE_THRESHOLD = 0.40  # Minimum confidence for detections
IMPORTANT_CLASSES = [0, 1, 2, 3, 5, 7]  # person, car, truck, bus, bicycle, motorbike

# Display settings
DISPLAY_WINDOW_NAME = "JARVIS-lite: Real-Time Vision + LLM"
MAX_DISPLAY_DETECTIONS = 20
```

### Environment Variables

**Required:**
```bash
# OpenRouter API key for LLM reasoning
export OPENROUTER_API_KEY="your_key_here"  # Linux/Mac
set OPENROUTER_API_KEY=your_key_here       # Windows CMD
$env:OPENROUTER_API_KEY="your_key_here"    # Windows PowerShell
```

**Optional:**
```bash
# Override default model (must be free)
export OPENROUTER_MODEL="mistralai/mistral-7b-instruct:free"
```

**Available Free Models:**
- `meta-llama/llama-3.3-70b-instruct:free` (default, best quality)
- `mistralai/mistral-7b-instruct:free` (fast, good quality)
- `google/gemma-7b-it:free` (Google model)
- `nousresearch/nous-capybara-7b:free` (alternative)

## Usage

### Basic Usage

Run the complete JARVIS-lite system:

```bash
# Set your API key first
export OPENROUTER_API_KEY="your_key_here"  # Linux/Mac
# OR
set OPENROUTER_API_KEY=your_key_here       # Windows CMD
# OR
$env:OPENROUTER_API_KEY="your_key_here"    # Windows PowerShell

# Run the system
python -m object_analytics.main
```

### What Happens

The system will:
1. Initialize YOLOv8 object detector
2. Connect to OpenRouter LLM (free model)
3. Load Vosk voice recognition model
4. Open your webcam
5. Start vision processing loop (30+ FPS)
6. Start voice interaction thread (parallel)
7. Display annotated video with FPS metrics
8. Listen for voice commands continuously
9. Respond with intelligent, scene-grounded answers

Press **Ctrl+C** or **Q** to exit gracefully.

### Expected Output

The system will display:
- **Live video feed** with bounding boxes and labels
- **Real-time FPS** and latency metrics
- **Detection confidence scores**
- **Voice interaction status** in console

**Console Output:**
```
INFO - Initializing JARVIS-lite Multimodal AI System...
INFO - LLM reasoning engine initialized successfully
INFO - Voice engine initialized successfully
INFO - 🎤 JARVIS voice interaction loop started
INFO - 📡 Using LLM-powered reasoning via OpenRouter
INFO - Vision processing: 28.5 FPS
INFO - User: "What do you see?"
INFO - JARVIS: "I see 2 objects: a person with 85% confidence and a car with 92% confidence."
```

## Voice Commands

JARVIS-lite understands natural language questions about the current scene.

### Scene Overview
- **"What do you see?"** → Complete list of detected objects
- **"How many objects?"** → Total count
- **"Describe the scene"** → Detailed scene analysis

### Object Queries
- **"What is the most important object?"** → Highest priority object
- **"Tell me about the person"** → Person-specific details
- **"Is there a car?"** → Presence detection

### Spatial Queries
- **"What's in the center?"** → Center-focused analysis
- **"Where is the main object?"** → Position description
- **"Is the path clear?"** → Safety assessment

### Context-Aware Queries
- **"Should I pay attention to something?"** → Priority advice
- **"What should I focus on?"** → Important object guidance
- **"Is anything important happening?"** → Event detection

The LLM provides intelligent, grounded answers based only on current detections.

### Performance Metrics

**Typical Performance (YOLOv8n on CPU):**
- **Vision FPS**: 25-30 fps (maintained during voice interaction)
- **LLM Latency**: 1-3 seconds (via OpenRouter free tier)
- **Voice Recognition**: ~500ms (Vosk offline)
- **Memory Usage**: ~2.5GB RAM (includes LLM client)

**Performance Factors:**
- **Hardware**: GPU acceleration significantly improves FPS
- **Resolution**: Lower input resolution = higher FPS
- **Model Size**: Larger YOLO models = higher accuracy, lower FPS
- **LLM Model**: Different free models have different latencies
- **Network**: LLM calls require internet (vision/voice work offline)

## API Reference

### LLMEngine(api_key, model, max_tokens, temperature)

LLM-powered reasoning engine using OpenRouter API.

**Parameters:**
- `api_key` (str): OpenRouter API key (or set OPENROUTER_API_KEY env variable)
- `model` (str): Model identifier (default: meta-llama/llama-3.3-70b-instruct:free)
- `max_tokens` (int): Maximum response length (default: 150)
- `temperature` (float): Sampling temperature (default: 0.3 for factual responses)

**Methods:**
- `answer(scene_summary, question, timeout)` → str: Generate grounded answer
- `test_connection()` → bool: Test OpenRouter API connectivity

**Example:**
```python
llm = LLMEngine()  # Uses OPENROUTER_API_KEY from environment
answer = llm.answer(summary, "What do you see?")
# Returns: "I see 2 objects: a person with 85% confidence and a car with 92% confidence."
```

### ObjectAnalytics.summarize(analytics_data)

Generates structured analytics summary optimized for LLM integration.

**Parameters:**
- `analytics_data` (list): List of analyzed detections

**Returns:**
- `dict`: Summary with total objects and complete detection list:
  ```python
  {
      'total_objects': int,
      'detections': [
          {
              'class': str,
              'confidence': float,
              'priority': float,
              'area_ratio': float,
              'position': str,
              'center': tuple,
              'bbox': list
          },
          ...
      ]
  }
  ```

**Example:**
```python
summary = analytics_engine.summarize(analyzed_detections)
# Returns complete scene data for LLM reasoning
```

### VoiceEngine(model_path)

Offline speech recognition and text-to-speech engine.

**Parameters:**
- `model_path` (str): Path to Vosk model directory

**Methods:**
- `listen()` → str: Captures and transcribes speech to text
- `speak(text)` → None: Converts text to speech output

**Example:**
```python
voice = VoiceEngine("models/vosk-model-small-en-us-0.15")
question = voice.listen()  # Waits for speech
voice.speak("Hello, I can see objects in the scene")
```

### VideoAnalyticsApp()

Main application class combining vision, analytics, LLM reasoning, and voice.

**Methods:**
- `run()` → int: Starts the complete multimodal system

**Example:**
```python
app = VideoAnalyticsApp()
exit_code = app.run()  # Full JARVIS-lite experience
```

## Multimodal Architecture

JARVIS-lite combines three parallel processing pipelines:

### Vision Pipeline (Main Thread)
- **Real-time Detection**: YOLOv8 processes video frames at 30+ FPS
- **Object Analytics**: Extracts confidence scores, positions, and priorities
- **Scene Summary**: Maintains latest detection state for LLM reasoning

### Voice Pipeline (Parallel Thread)
- **Speech Recognition**: Vosk processes audio input offline
- **Text-to-Speech**: pyttsx3 generates natural voice responses
- **Non-blocking Operation**: Runs independently to preserve vision FPS

### Reasoning Pipeline (On-Demand)
- **LLM Integration**: OpenRouter API with free models
- **Question Processing**: Natural language understanding of user queries
- **Scene Analysis**: LLM interprets structured scene data
- **Grounded Responses**: Factual answers based only on detections
- **Timeout Protection**: 5-second timeout prevents blocking

### Threading Model
```
Main Thread: Vision processing (30+ FPS) → Scene Summary
     ↓
Voice Thread: Speech I/O → LLM Reasoning → Answer (Non-blocking)
     ↓
LLM Call: OpenRouter API (1-3s latency, asynchronous)
```

**Critical Design Principles:**
1. Vision never waits for LLM or voice
2. Voice queries use latest scene snapshot
3. LLM timeout prevents indefinite blocks
4. Fallback reasoning when LLM unavailable
5. Graceful degradation at all levels

This architecture ensures smooth multimodal interaction while maintaining real-time performance.

## Troubleshooting

### LLM / OpenRouter Issues
- **"LLM functionality disabled"**: Set OPENROUTER_API_KEY environment variable
- **LLM timeout errors**: Network latency - LLM falls back to basic summaries
- **"Model not found"**: Check model name format (must include ":free" suffix)
- **Rate limiting**: OpenRouter free tier has limits - wait and retry

### Voice Recognition Issues
- **No microphone detected**: Ensure microphone is connected and enabled in system settings
- **Poor recognition accuracy**: Speak clearly and closer to microphone
- **Vosk model not found**: Verify model is extracted to `object_analytics/models/vosk-model-small-en-us-0.15/`

### Performance Issues
- **Low FPS**: Close other applications using camera/GPU
- **LLM lag**: Normal for free tier (1-3s) - doesn't affect vision FPS
- **Audio lag**: Voice processing runs in background thread - should not affect vision
- **Memory usage**: YOLOv8 + LLM client requires ~2.5GB RAM

### Installation Issues
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Python version**: Requires Python 3.12+ for optimal performance
- **OpenCV errors**: Install system dependencies: `pip install opencv-python`
- **openai library errors**: Update to latest: `pip install --upgrade openai`

### Common Errors
- **"Model not found"**: Download and extract Vosk model to correct directory
- **"No audio device"**: Check microphone permissions and connections
- **"CUDA out of memory"**: Use CPU mode or reduce model size
- **"OpenRouter connection failed"**: Check internet connection and API key

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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