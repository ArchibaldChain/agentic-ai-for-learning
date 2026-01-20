"""
Abstract LLM Client Base Class.

This module defines the abstract interface for LLM clients, allowing
the agentic AI system to work with multiple providers (Bedrock, Azure OpenAI,
OpenAI, etc.) through a unified interface.

The key design principle is that agents should not need to change when
switching LLM providers - they just call the client interface, and the
concrete implementation handles provider-specific details.

Standardized Response Formats:

For generate():
    {
        "content": str,              # Generated text
        "model": str,                # Model identifier
        "usage": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        },
        "finish_reason": str,        # "stop", "length", "tool_use", etc.
        "raw_response": dict         # Original provider response
    }

For generate_with_tools():
    {
        "content": str | None,       # Generated text (may be None if tool call)
        "tool_calls": [              # List of tool calls (empty if none)
            {
                "id": str,           # Unique tool call ID
                "name": str,         # Tool name
                "input": dict        # Tool input arguments
            }
        ],
        "finish_reason": str,        # "stop", "tool_use", etc.
        "model": str,
        "usage": {...},
        "raw_response": dict
    }

For get_embeddings():
    [[float, ...], ...]              # List of embedding vectors

Example usage:
    from agentic_ai.llm import BedrockClient, AzureClient

    # Use Bedrock
    client = BedrockClient()
    response = client.generate([{"role": "user", "content": "Hello"}])

    # Swap to Azure (when implemented)
    client = AzureClient()
    response = client.generate([{"role": "user", "content": "Hello"}])
    # Same response format!
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    Standardized message format for LLM conversations.

    Attributes:
        role: The message role ("system", "user", "assistant", "tool")
        content: The message content (text or structured content)
        name: Optional name for tool messages
        tool_call_id: Optional ID linking to a tool call
    """
    role: str
    content: str | list[dict[str, Any]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ToolDefinition:
    """
    Standardized tool definition for LLM tool calling.

    Attributes:
        name: Unique tool identifier
        description: What the tool does (used by LLM to decide when to use it)
        parameters: JSON Schema for tool parameters
    """
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    This class defines the interface that all LLM provider implementations
    must follow. The goal is to provide a consistent API across different
    providers so that agent code doesn't need to change when switching
    between providers.

    Subclasses must implement:
    - generate(): Basic text generation
    - generate_with_tools(): Generation with tool/function calling
    - get_embeddings(): Text embeddings

    Optionally override:
    - stream(): Streaming responses (default raises NotImplementedError)

    Example:
        class MyProvider(LLMClient):
            def generate(self, messages, **kwargs):
                # Provider-specific implementation
                return standardized_response
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                      Roles: "user", "assistant", "system"
            system_prompt: Optional system prompt (some providers handle separately)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            **kwargs: Provider-specific options

        Returns:
            Standardized response dict:
            {
                "content": str,
                "model": str,
                "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
                "finish_reason": str,
                "raw_response": dict
            }

        Raises:
            Exception: Provider-specific errors
        """
        pass

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate a completion with tool calling support.

        Args:
            messages: List of message dicts
            tools: List of tool definitions, each with:
                   - name: Tool identifier
                   - description: What the tool does
                   - parameters: JSON Schema for inputs (or input_schema)
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options

        Returns:
            Standardized response dict:
            {
                "content": str | None,
                "tool_calls": [{"id": str, "name": str, "input": dict}, ...],
                "finish_reason": str,
                "model": str,
                "usage": {...},
                "raw_response": dict
            }
        """
        pass

    @abstractmethod
    def get_embeddings(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[list[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            **kwargs: Provider-specific options (model, dimensions, etc.)

        Returns:
            List of embedding vectors, one per input text.
            Each vector is a list of floats.

        Example:
            >>> embeddings = client.get_embeddings(["Hello", "World"])
            >>> len(embeddings)
            2
            >>> len(embeddings[0])  # Dimension varies by model
            1536
        """
        pass

    def stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream a completion token by token.

        This method is optional - subclasses that support streaming
        should override this method.

        Args:
            messages: List of message dicts
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options

        Yields:
            String chunks as they are generated.

        Raises:
            NotImplementedError: If streaming is not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming"
        )

    @staticmethod
    def format_messages(
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """
        Helper to format messages for the generate methods.

        This is a convenience method for simple use cases where you
        just have a user message and optionally a system prompt.

        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            conversation_history: Optional previous messages to include

        Returns:
            List of message dicts ready for generate()

        Example:
            >>> messages = LLMClient.format_messages(
            ...     "What is AI?",
            ...     system_prompt="You are a helpful assistant."
            ... )
            >>> response = client.generate(messages)
        """
        messages = []

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add the user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        return messages

    @staticmethod
    def format_tool_result(
        tool_call_id: str,
        result: str,
    ) -> dict[str, Any]:
        """
        Helper to format a tool result message.

        After executing a tool, use this to format the result
        for sending back to the LLM.

        Args:
            tool_call_id: The ID from the tool call
            result: The result from executing the tool

        Returns:
            Message dict ready to append to messages list
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }


def extract_text_content(response: dict[str, Any]) -> str:
    """
    Extract text content from a standardized response.

    Args:
        response: Response dict from generate() or generate_with_tools()

    Returns:
        The text content, or empty string if none.
    """
    content = response.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Handle structured content (list of content blocks)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    return str(content)


def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract tool calls from a standardized response.

    Args:
        response: Response dict from generate_with_tools()

    Returns:
        List of tool call dicts with "id", "name", and "input" keys.
        Empty list if no tool calls.
    """
    return response.get("tool_calls", [])


def has_tool_calls(response: dict[str, Any]) -> bool:
    """
    Check if a response contains tool calls.

    Args:
        response: Response dict from generate_with_tools()

    Returns:
        True if response contains at least one tool call.
    """
    return len(extract_tool_calls(response)) > 0
