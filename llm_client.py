from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    GEMINI_TOOL_DEFINITIONS,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
)

# Configure logging
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM fails."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid."""
    pass


class ChatResponse:
    """Structured response from LLM chat."""
    
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop"
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason
        }


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: list[dict], use_tools: bool = True) -> dict[str, Any]:
        """
        Send a chat request to the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            use_tools: Whether to enable tool/function calling
            
        Returns:
            A dict with:
                - 'content': The text response (may be None if tool call)
                - 'tool_calls': List of tool calls (if any)
                - 'finish_reason': Why the response ended
                
        Raises:
            LLMConnectionError: If connection to LLM fails
            LLMResponseError: If response is invalid
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass


class GroqClient(LLMClient):
    """Groq LLM client with function calling support."""
    
    def __init__(self, api_key: str, model: str | None = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model name (defaults to config value)
            
        Raises:
            LLMConnectionError: If client initialization fails
        """
        if not api_key or not api_key.strip():
            raise LLMConnectionError("API key is required")
        
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key.strip())
            self.model = model or DEFAULT_GROQ_MODEL
            logger.info(f"Groq client initialized with model: {self.model}")
        except ImportError:
            raise LLMConnectionError("groq package not installed. Run: pip install groq")
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize Groq client: {e}")
    
    @property
    def provider_name(self) -> str:
        return "groq"
    
    def chat(self, messages: list[dict], use_tools: bool = True) -> dict[str, Any]:
        """Send a chat request to Groq."""
        # Prepare messages with system prompt
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
        }
        
        if use_tools:
            kwargs["tools"] = TOOL_DEFINITIONS
            kwargs["tool_choice"] = "auto"
        
        try:
            logger.debug(f"Sending request to Groq with {len(messages)} messages")
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise LLMConnectionError(f"Failed to communicate with Groq: {e}")
        
        if not response.choices:
            raise LLMResponseError("Empty response from Groq")
        
        message = response.choices[0].message
        result = ChatResponse(
            content=message.content,
            finish_reason=response.choices[0].finish_reason
        )
        
        # Extract tool calls if present
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in tool arguments: {tool_call.function.arguments}")
                    arguments = {}
                
                result.tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": arguments
                })
        
        logger.debug(f"Groq response: content={bool(result.content)}, tool_calls={len(result.tool_calls)}")
        return result.to_dict()


class GeminiClient(LLMClient):
    """Google Gemini LLM client with function calling support."""
    
    def __init__(self, api_key: str, model: str | None = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google AI API key
            model: Model name (defaults to config value)
            
        Raises:
            LLMConnectionError: If client initialization fails
        """
        if not api_key or not api_key.strip():
            raise LLMConnectionError("API key is required")
        
        try:
            import google.generativeai as genai
            self._genai = genai
            genai.configure(api_key=api_key.strip())
            
            model_name = model or DEFAULT_GEMINI_MODEL
            
            # Create tool declarations
            self.tools = self._create_tools()
            
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
                tools=self.tools
            )
            self.model_no_tools = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT
            )
            self._model_name = model_name
            logger.info(f"Gemini client initialized with model: {model_name}")
        except ImportError:
            raise LLMConnectionError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize Gemini client: {e}")
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    def _create_tools(self) -> list:
        """Create Gemini-compatible tool declarations."""
        genai = self._genai
        
        function_declarations = []
        for tool_def in GEMINI_TOOL_DEFINITIONS:
            properties = {}
            for k, v in tool_def["parameters"].get("properties", {}).items():
                properties[k] = genai.protos.Schema(
                    type=self._get_gemini_type(v.get("type", "string")),
                    description=v.get("description", "")
                )
            
            func_decl = genai.protos.FunctionDeclaration(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties=properties,
                    required=tool_def["parameters"].get("required", [])
                )
            )
            function_declarations.append(func_decl)
        
        return [genai.protos.Tool(function_declarations=function_declarations)]
    
    def _get_gemini_type(self, type_str: str):
        """Convert JSON schema type to Gemini type."""
        genai = self._genai
        
        type_map = {
            "string": genai.protos.Type.STRING,
            "number": genai.protos.Type.NUMBER,
            "integer": genai.protos.Type.INTEGER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT,
        }
        return type_map.get(type_str, genai.protos.Type.STRING)
    
    def _convert_messages_to_gemini(self, messages: list[dict]) -> list:
        """Convert OpenAI-style messages to Gemini format."""
        genai = self._genai
        gemini_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [content] if content else [""]
                })
            elif role == "assistant":
                parts = []
                if content:
                    parts.append(content)
                # Handle function calls in assistant messages
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        parts.append(genai.protos.Part(
                            function_call=genai.protos.FunctionCall(
                                name=tc["name"],
                                args=tc.get("arguments", {})
                            )
                        ))
                if parts:
                    gemini_messages.append({
                        "role": "model",
                        "parts": parts
                    })
            elif role == "tool":
                # Tool response
                gemini_messages.append({
                    "role": "user",
                    "parts": [genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=msg.get("name", "unknown"),
                            response={"result": content}
                        )
                    )]
                })
        
        return gemini_messages
    
    def chat(self, messages: list[dict], use_tools: bool = True) -> dict[str, Any]:
        """Send a chat request to Gemini."""
        gemini_messages = self._convert_messages_to_gemini(messages)
        model = self.model if use_tools else self.model_no_tools
        
        try:
            logger.debug(f"Sending request to Gemini with {len(messages)} messages")
            response = model.generate_content(gemini_messages)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise LLMConnectionError(f"Failed to communicate with Gemini: {e}")
        
        result = ChatResponse()
        
        # Extract response
        if response.candidates:
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        result.content = part.text
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        result.tool_calls.append({
                            "id": f"call_{fc.name}",
                            "name": fc.name,
                            "arguments": dict(fc.args) if fc.args else {}
                        })
                        result.finish_reason = "tool_calls"
        
        logger.debug(f"Gemini response: content={bool(result.content)}, tool_calls={len(result.tool_calls)}")
        return result.to_dict()


def create_client(provider: str, api_key: str) -> LLMClient:
    """
    Create an LLM client based on the provider.
    
    Args:
        provider: Either "groq" or "gemini"
        api_key: The API key for the provider
        
    Returns:
        An LLM client instance
        
    Raises:
        ValueError: If provider is unknown
        LLMConnectionError: If client initialization fails
    """
    provider = provider.lower().strip()
    
    if provider == "groq":
        return GroqClient(api_key)
    elif provider == "gemini":
        return GeminiClient(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: 'groq', 'gemini'")
