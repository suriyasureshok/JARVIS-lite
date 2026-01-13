"""
Voice Engine Module for JARVIS-lite.

This module provides offline speech-to-text and text-to-speech capabilities
for multimodal interaction with the Real-Time Object Analytics Engine.
"""

import queue
import sounddevice as sd
import json
import pyttsx3
from vosk import Model, KaldiRecognizer
import logging

logger = logging.getLogger(__name__)


class VoiceEngine:
    """
    Voice engine for speech recognition and synthesis.

    Provides offline speech-to-text using Vosk and text-to-speech using pyttsx3.
    Designed for low-latency, non-blocking operation in multimodal systems.
    """

    def __init__(self, model_path: str):
        """
        Initialize the voice engine.

        Parameters
        ----------
        model_path : str
            Path to the Vosk model directory.

        Raises
        ------
        RuntimeError
            If model loading fails.
        """
        try:
            logger.info(f"Loading Vosk model from: {model_path}")
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.q = queue.Queue()

            logger.info("Initializing TTS engine")
            self.engine = pyttsx3.init()

            # Configure TTS for better performance
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break

            self.engine.setProperty('rate', 180)  # Slightly faster than default
            self.engine.setProperty('volume', 0.9)

            logger.info("Voice engine initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize voice engine: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def callback(self, indata, frames, time, status):
        """
        Audio callback for sounddevice input stream.

        Parameters
        ----------
        indata : numpy.ndarray
            Audio data from microphone.
        frames : int
            Number of frames.
        time : float
            Current time.
        status : str
            Status information.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.q.put(bytes(indata))

    def listen(self) -> str:
        """
        Listen for speech input and convert to text.

        Blocks until speech is detected and recognized.

        Returns
        -------
        str
            Recognized text from speech input.

        Raises
        ------
        RuntimeError
            If speech recognition fails.
        """
        try:
            logger.info("🎤 Listening for speech input...")

            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                   channels=1, callback=self.callback):
                while True:
                    data = self.q.get()
                    if self.recognizer.AcceptWaveform(data):
                        result = self.recognizer.Result()
                        text = json.loads(result).get("text", "").strip()

                        if text:
                            logger.info(f"🎤 Recognized: '{text}'")
                            return text

        except Exception as e:
            error_msg = f"Speech recognition failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def speak(self, text: str) -> None:
        """
        Convert text to speech and output through speakers.

        Parameters
        ----------
        text : str
            Text to convert to speech.

        Raises
        ------
        RuntimeError
            If text-to-speech fails.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to speak()")
            return

        try:
            logger.info(f"🔊 Speaking: '{text}'")
            self.engine.say(text)
            self.engine.runAndWait()
            logger.debug("Speech output completed")

        except Exception as e:
            error_msg = f"Text-to-speech failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def set_voice_properties(self, rate: int = None, volume: float = None) -> None:
        """
        Configure voice properties.

        Parameters
        ----------
        rate : int, optional
            Speech rate (words per minute).
        volume : float, optional
            Volume level (0.0 to 1.0).
        """
        if rate is not None:
            self.engine.setProperty('rate', rate)
            logger.info(f"Voice rate set to: {rate}")

        if volume is not None:
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
            logger.info(f"Voice volume set to: {volume}")