from __future__ import annotations

import json
import logging
from typing import Any

from config import MAX_CHAT_ITERATIONS
from llm_client import LLMClient, LLMError, create_client
from tools import add_to_history, execute_tool, reset_session

# Configure logging
logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class ExaminerAgent:
    """
    AI Examiner Agent that conducts NLP exams.
    
    Manages conversation history, handles tool calls, and coordinates
    with the LLM to conduct the exam.
    """
    
    def __init__(self, provider: str, api_key: str) -> None:
        """
        Initialize the agent.
        
        Args:
            provider: LLM provider ("groq" or "gemini")
            api_key: API key for the LLM provider
            
        Raises:
            AgentError: If initialization fails
        """
        try:
            self.client: LLMClient = create_client(provider, api_key)
            self.messages: list[dict[str, Any]] = []
            self.provider = provider.lower()
            logger.info(f"ExaminerAgent initialized with provider: {provider}")
        except (ValueError, LLMError) as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise AgentError(str(e))
    
    def reset(self) -> None:
        """Reset the agent state for a new exam."""
        self.messages = []
        reset_session()
        logger.info("Agent state reset")
    
    def _process_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """
        Process tool calls and return results.
        
        Args:
            tool_calls: List of tool calls from the LLM
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_id = tool_call.get("id", f"call_{tool_name}")
            arguments = tool_call.get("arguments", {})
            
            logger.debug(f"Processing tool call: {tool_name}")
            
            # Execute the tool
            result = execute_tool(tool_name, arguments)
            
            # Store tool call and result in messages based on provider
            if self.provider == "groq":
                self._add_groq_tool_messages(tool_id, tool_name, arguments, result)
            else:
                self._add_gemini_tool_messages(tool_id, tool_name, arguments, result)
            
            results.append(result)
        
        return results
    
    def _add_groq_tool_messages(
        self,
        tool_id: str,
        tool_name: str,
        arguments: dict,
        result: dict
    ) -> None:
        """Add tool call messages in Groq format."""
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments)
                }
            }]
        })
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": json.dumps(result)
        })
    
    def _add_gemini_tool_messages(
        self,
        tool_id: str,
        tool_name: str,
        arguments: dict,
        result: dict
    ) -> None:
        """Add tool call messages in Gemini format."""
        self.messages.append({
            "role": "assistant",
            "content": f"[Calling {tool_name}]",
            "tool_calls": [{
                "id": tool_id,
                "name": tool_name,
                "arguments": arguments
            }]
        })
        self.messages.append({
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(result)
        })
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: The user's message
            
        Returns:
            The agent's response
            
        Raises:
            AgentError: If chat processing fails critically
        """
        if not user_message or not user_message.strip():
            return "Please provide a message."
        
        user_message = user_message.strip()
        
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        add_to_history("user", user_message)
        
        logger.debug(f"Processing user message: {user_message[:50]}...")
        
        # Keep processing until we get a final response
        iteration = 0
        
        while iteration < MAX_CHAT_ITERATIONS:
            iteration += 1
            
            try:
                response = self.client.chat(self.messages)
            except LLMError as e:
                logger.error(f"LLM error during chat: {e}")
                error_msg = f"Communication error: {e}"
                self.messages.append({"role": "assistant", "content": error_msg})
                return error_msg
            
            # Check if there are tool calls to process
            if response.get("tool_calls"):
                logger.debug(f"Processing {len(response['tool_calls'])} tool calls")
                self._process_tool_calls(response["tool_calls"])
                continue
            
            # No tool calls, we have the final response
            content = response.get("content")
            if content:
                self.messages.append({
                    "role": "assistant",
                    "content": content
                })
                add_to_history("assistant", content)
                logger.debug(f"Agent response: {content[:50]}...")
                return content
            
            # Empty response, break to avoid infinite loop
            logger.warning("Empty response from LLM")
            break
        
        if iteration >= MAX_CHAT_ITERATIONS:
            logger.warning(f"Max iterations ({MAX_CHAT_ITERATIONS}) reached")
        
        fallback = "I apologize, but I encountered an issue processing your request. Please try again."
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback
    
    def get_initial_greeting(self) -> str:
        """
        Get the initial greeting from the agent.
        
        Returns:
            The agent's greeting message
        """
        logger.debug("Getting initial greeting")
        
        try:
            response = self.client.chat(self.messages)
            
            if response.get("content"):
                greeting = response["content"]
                self.messages.append({
                    "role": "assistant",
                    "content": greeting
                })
                add_to_history("assistant", greeting)
                logger.debug("Initial greeting obtained from LLM")
                return greeting
        except LLMError as e:
            logger.error(f"Error getting greeting: {e}")
        
        # Fallback greeting
        fallback = (
            "Hello! I'm your AI Examiner for the NLP course. "
            "To begin the exam, please provide your name and email address."
        )
        self.messages.append({
            "role": "assistant",
            "content": fallback
        })
        add_to_history("assistant", fallback)
        logger.debug("Using fallback greeting")
        return fallback
