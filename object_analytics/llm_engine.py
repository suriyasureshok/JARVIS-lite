"""
LLM Engine Module for JARVIS-lite.

This module provides intelligent reasoning capabilities using Large Language Models
via OpenRouter API. It processes structured scene analytics and answers user questions
with grounded, factual responses based only on detected objects.

Architecture:
- Uses OpenRouter API for free LLM access (meta-llama/llama-3.3-70b-instruct)
- Implements strict prompt engineering to prevent hallucination
- Operates on structured scene data only (no raw images)
- Provides concise, context-aware answers

Author: JARVIS-lite Development Team
License: MIT
"""

import os
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMEngine:
    """
    LLM-powered reasoning engine using OpenRouter API.
    
    This class provides intelligent question answering capabilities by processing
    structured scene analytics through a free LLM. It ensures responses are grounded
    in actual detections and never hallucinates information.
    
    Attributes:
        client (OpenAI): OpenRouter API client
        model (str): Free model identifier
        system_prompt (str): Prompt template enforcing grounded reasoning
        max_tokens (int): Maximum response length
        temperature (float): Sampling temperature for response generation
    
    Examples:
        >>> llm = LLMEngine(api_key="your_openrouter_key")
        >>> scene = {
        ...     'total_objects': 2,
        ...     'detections': [
        ...         {'class': 'person', 'confidence': 0.85, 'priority': 8.5},
        ...         {'class': 'car', 'confidence': 0.92, 'priority': 6.3}
        ...     ]
        ... }
        >>> answer = llm.answer(scene, "What is the most important object?")
        >>> print(answer)
        'The most important object is a person with 85% confidence and high priority score.'
    """
    
    # Default free model from OpenRouter
    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
    
    # Alternative free models (fallback options)
    FALLBACK_MODELS = [
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-3-27b-it:free",
        "nousresearch/nous-capybara-7b:free"
    ]
    
    # OpenRouter API endpoint
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.3
    ):
        """
        Initialize the LLM reasoning engine.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env variable)
            model: Model identifier (defaults to free llama-3.3-70b)
            max_tokens: Maximum tokens in response (default: 150 for concise answers)
            temperature: Sampling temperature (default: 0.3 for factual responses)
        
        Raises:
            ValueError: If API key is not provided via argument or environment
            RuntimeError: If OpenRouter client initialization fails
        """
        # Get API key from argument or environment
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        
        if not self.api_key:
            logger.warning(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter. LLM functionality will be disabled."
            )
            self.client = None
            self.model = None
            return
        
        # Initialize OpenRouter client (OpenAI-compatible API)
        try:
            self.client = OpenAI(
                base_url=self.OPENROUTER_BASE_URL,
                api_key=self.api_key
            )
            logger.info("OpenRouter client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.client = None
            raise RuntimeError(f"OpenRouter initialization failed: {e}")
        
        # Set model and generation parameters
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # System prompt enforcing grounded reasoning
        self.system_prompt = self._build_system_prompt()
        
        logger.info(f"LLM Engine initialized with model: {self.model}")
        logger.info(f"Parameters: max_tokens={max_tokens}, temperature={temperature}")
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt enforcing grounded, factual responses.
        
        This prompt is critical for preventing hallucination and ensuring the LLM
        only uses information from the provided scene data.
        
        Returns:
            str: System prompt template
        """
        return """You are JARVIS, a real-time multimodal AI assistant with computer vision capabilities.

CRITICAL RULES:
1. You can ONLY see what the computer vision system detects in the current frame
2. NEVER make up or hallucinate information about objects not in the scene data
3. If asked about something not detected, respond: "I don't see that in the current frame"
4. Be concise - respond in 1-2 sentences maximum
5. Use natural language, not technical jargon
6. Reference confidence scores when relevant to show certainty

SCENE DATA FORMAT:
You receive structured analytics with:
- total_objects: Number of detected objects
- detections: List of objects with class, confidence, position, area_ratio, priority

RESPONSE STYLE:
- Professional but conversational
- Fact-based and precise
- Helpful and actionable
- Never uncertain or vague about what you see

EXAMPLE EXCHANGES:
User: "What do you see?"
You: "I see 2 objects: a person with 85% confidence and a car with 92% confidence."

User: "Is there a dog?"
You: "I don't see a dog in the current frame."

User: "What should I pay attention to?"
You: "The person has the highest priority score, likely due to proximity and size."

Now answer based ONLY on the scene data provided."""
    
    def answer(
        self,
        scene_summary: Dict[str, Any],
        question: str,
        timeout: float = 5.0
    ) -> str:
        """
        Answer a user question based on structured scene analytics.
        
        This method sends the scene summary and question to the LLM, which generates
        a grounded, factual response. The system prompt ensures the LLM never invents
        information and only references detected objects.
        
        Args:
            scene_summary: Structured analytics from ObjectAnalytics.summarize()
                Expected format: {
                    'total_objects': int,
                    'detections': [
                        {
                            'class': str,
                            'confidence': float,
                            'position': tuple,
                            'area_ratio': float,
                            'priority': float
                        },
                        ...
                    ]
                }
            question: User's natural language question
            timeout: Maximum seconds to wait for LLM response (default: 5.0)
        
        Returns:
            str: LLM-generated answer grounded in scene data
        
        Raises:
            RuntimeError: If LLM client is not initialized
            TimeoutError: If LLM takes longer than timeout
        
        Examples:
            >>> scene = {'total_objects': 1, 'detections': [{'class': 'person', ...}]}
            >>> answer = llm.answer(scene, "What do you see?")
            >>> print(answer)
            'I see 1 object: a person with high confidence.'
        """
        # Check if LLM is available
        if self.client is None:
            return (
                "LLM functionality is disabled. Please set OPENROUTER_API_KEY "
                "environment variable to enable intelligent reasoning."
            )
        
        # Validate scene summary
        if not scene_summary or 'total_objects' not in scene_summary:
            return "No scene data available. Camera may be disconnected."
        
        # Build user message with structured scene context
        user_message = self._format_scene_context(scene_summary, question)
        
        # Call LLM with timeout protection
        try:
            logger.info(f"Sending question to LLM: '{question}'")
            logger.debug(f"Scene context: {scene_summary}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=timeout
            )
            
            # Extract answer from response
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM response: '{answer}'")
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            
            # Fallback to basic scene description on error
            return self._fallback_answer(scene_summary, question)
    
    def _format_scene_context(self, scene_summary: Dict[str, Any], question: str) -> str:
        """
        Format scene data and question into LLM input.
        
        Args:
            scene_summary: Structured analytics dictionary
            question: User's question
        
        Returns:
            str: Formatted context string
        """
        # Extract detection data
        total = scene_summary.get('total_objects', 0)
        detections = scene_summary.get('detections', [])
        
        # Build concise scene description
        if total == 0:
            scene_text = "CURRENT SCENE: Empty (no objects detected)"
        else:
            scene_text = f"CURRENT SCENE: {total} object(s) detected\n\nDETECTIONS:\n"
            for i, det in enumerate(detections[:10], 1):  # Limit to top 10
                scene_text += (
                    f"{i}. {det.get('class', 'unknown')} - "
                    f"Confidence: {det.get('confidence', 0):.0%}, "
                    f"Priority: {det.get('priority', 0):.1f}/10, "
                    f"Area: {det.get('area_ratio', 0):.1%}, "
                    f"Position: {det.get('position', 'unknown')}\n"
                )
        
        # Combine scene context with question
        return f"{scene_text}\n\nUSER QUESTION: {question}\n\nYOUR ANSWER:"
    
    def _fallback_answer(self, scene_summary: Dict[str, Any], question: str) -> str:
        """
        Provide fallback answer when LLM fails.
        
        Uses simple rule-based logic to answer common questions without LLM.
        
        Args:
            scene_summary: Scene analytics
            question: User question
        
        Returns:
            str: Fallback answer
        """
        total = scene_summary.get('total_objects', 0)
        detections = scene_summary.get('detections', [])
        
        question_lower = question.lower()
        
        # Basic question patterns
        if 'how many' in question_lower or 'count' in question_lower:
            if total == 0:
                return "I don't see any objects right now."
            return f"I see {total} object{'s' if total != 1 else ''} in the scene."
        
        elif 'what' in question_lower and ('see' in question_lower or 'detect' in question_lower):
            if total == 0:
                return "I don't see anything right now."
            classes = [d.get('class', 'unknown') for d in detections[:3]]
            return f"I see: {', '.join(classes)}."
        
        elif 'important' in question_lower or 'priority' in question_lower:
            if total == 0:
                return "No objects detected to prioritize."
            top = detections[0] if detections else None
            if top:
                return (
                    f"The most important object is a {top.get('class', 'object')} "
                    f"with {top.get('confidence', 0):.0%} confidence."
                )
        
        # Generic fallback
        return f"LLM service unavailable. Current scene has {total} detected objects."
    
    def test_connection(self) -> bool:
        """
        Test OpenRouter API connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.client is None:
            logger.warning("LLM client not initialized")
            return False
        
        try:
            # Simple test query
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Respond with OK if you're working."}
                ],
                max_tokens=10,
                timeout=3.0
            )
            logger.info("OpenRouter connection test successful")
            return True
        except Exception as e:
            logger.error(f"OpenRouter connection test failed: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of LLM engine."""
        status = "active" if self.client else "disabled"
        return f"LLMEngine(model={self.model}, status={status})"


# Convenience function for quick usage
def create_llm_engine(api_key: Optional[str] = None) -> LLMEngine:
    """
    Factory function to create LLM engine instance.
    
    Args:
        api_key: Optional OpenRouter API key
    
    Returns:
        LLMEngine: Initialized engine instance
    
    Examples:
        >>> llm = create_llm_engine()
        >>> answer = llm.answer(scene_data, "What do you see?")
    """
    return LLMEngine(api_key=api_key)
