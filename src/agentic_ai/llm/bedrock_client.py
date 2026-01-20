"""
AWS Bedrock Client for Claude API calls.

This module provides a Bedrock-specific implementation of the LLMClient
abstract class for making calls to Claude models via AWS Bedrock.

It handles message formatting, tool use, streaming, and embeddings
using the Bedrock Converse API and Titan Embeddings.

Example usage:
    from agentic_ai.llm.bedrock_client import BedrockClient

    client = BedrockClient()

    # Simple message (using standardized interface)
    response = client.generate([
        {"role": "user", "content": "What is 2 + 2?"}
    ])
    print(response["content"])

    # With tools
    tools = [
        {
            "name": "calculator",
            "description": "Performs basic math operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    ]
    response = client.generate_with_tools(
        [{"role": "user", "content": "What is 15 * 7?"}],
        tools=tools
    )

    # Legacy interface (still supported)
    response = client.invoke("What is 2 + 2?")
"""

import json
import logging
from typing import Any, Generator, Optional

import boto3
from botocore.exceptions import ClientError

from .llm_client import LLMClient
from ..utils.config import get_config, Config

logger = logging.getLogger(__name__)


class BedrockClient(LLMClient):
    """
    AWS Bedrock implementation of LLMClient.

    This class wraps the boto3 Bedrock runtime client and provides
    both the standardized LLMClient interface and legacy methods
    for backward compatibility.

    The Bedrock API uses the "converse" API which provides a unified
    interface for different models. For Claude, it supports:
    - Multi-turn conversations with message history
    - Tool use (function calling)
    - System prompts
    - Streaming responses

    Attributes:
        client: boto3 Bedrock runtime client
        model_id: The Claude model ID to use
        embedding_model_id: The Titan model ID for embeddings
        config: Configuration object with AWS settings
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Bedrock client.

        Args:
            config: Optional Config object. If not provided, loads from environment.

        Example:
            >>> client = BedrockClient()
            >>> # Or with custom config
            >>> from agentic_ai.utils.config import Config
            >>> custom_config = Config(aws_region="us-west-2", ...)
            >>> client = BedrockClient(config=custom_config)
        """
        self.config = config or get_config()
        self.model_id = self.config.bedrock_model_id
        self.embedding_model_id = self.config.embedding_model_id

        # Create boto3 client for Bedrock runtime
        session_kwargs = {
            "region_name": self.config.aws_region,
        }

        # Only add credentials if explicitly provided
        if self.config.aws_access_key_id and self.config.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = self.config.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key

        self.client = boto3.client("bedrock-runtime", **session_kwargs)

        logger.info(f"Initialized BedrockClient with model: {self.model_id}")

    # =========================================================================
    # LLMClient Abstract Method Implementations
    # =========================================================================

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate a completion from Claude via Bedrock.

        This implements the standardized LLMClient interface.

        Args:
            messages: List of message dicts with "role" and "content" keys
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            **kwargs: Additional Bedrock-specific options

        Returns:
            Standardized response dict with content, model, usage, finish_reason
        """
        logger.debug(f"generate() called with {len(messages)} messages")

        # Convert messages to Bedrock format
        bedrock_messages = self._convert_messages_to_bedrock(messages)

        # Build request parameters
        request_params = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            }
        }

        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]

        try:
            response = self.client.converse(**request_params)
            return self._convert_response_to_standard(response)

        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise

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
            tools: List of tool definitions with name, description, parameters
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional options

        Returns:
            Standardized response with content, tool_calls, finish_reason, etc.
        """
        logger.debug(f"generate_with_tools() called with {len(tools)} tools")

        # Convert messages and tools to Bedrock format
        bedrock_messages = self._convert_messages_to_bedrock(messages)
        bedrock_tools = self._convert_tools_to_bedrock(tools)

        # Build request
        request_params = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
            "toolConfig": {"tools": bedrock_tools}
        }

        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]

        try:
            response = self.client.converse(**request_params)
            return self._convert_response_to_standard_with_tools(response)

        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise

    def get_embeddings(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[list[float]]:
        """
        Get embeddings for texts using Bedrock Titan Embeddings.

        Args:
            texts: List of texts to embed
            **kwargs: Additional options (normalize, etc.)

        Returns:
            List of embedding vectors
        """
        logger.debug(f"get_embeddings() called with {len(texts)} texts")

        normalize = kwargs.get("normalize", True)
        embeddings = []

        for text in texts:
            embedding = self._get_single_embedding(text, normalize=normalize)
            embeddings.append(embedding)

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

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

        Args:
            messages: List of message dicts
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional options

        Yields:
            String chunks as they are generated
        """
        bedrock_messages = self._convert_messages_to_bedrock(messages)

        request_params = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            }
        }

        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]

        try:
            response = self.client.converse_stream(**request_params)

            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        yield delta["text"]

        except ClientError as e:
            logger.error(f"Bedrock streaming error: {e}")
            raise

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _convert_messages_to_bedrock(
        self,
        messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert standardized messages to Bedrock format.

        Bedrock expects:
        - role: "user" or "assistant"
        - content: list of content blocks [{"text": "..."}]

        For tool results, Bedrock expects:
        - role: "user"
        - content: [{"toolResult": {"toolUseId": "...", "content": [{"text": "..."}]}}]
        """
        bedrock_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle tool result messages
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                bedrock_messages.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": tool_call_id,
                            "content": [{"text": str(content)}]
                        }
                    }]
                })
            # Handle regular messages
            elif isinstance(content, str):
                bedrock_messages.append({
                    "role": role,
                    "content": [{"text": content}]
                })
            elif isinstance(content, list):
                # Content is already in block format
                bedrock_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                bedrock_messages.append({
                    "role": role,
                    "content": [{"text": str(content)}]
                })

        return bedrock_messages

    def _convert_tools_to_bedrock(
        self,
        tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert standardized tool definitions to Bedrock format.

        Bedrock expects:
        {
            "toolSpec": {
                "name": "...",
                "description": "...",
                "inputSchema": {"json": {...}}
            }
        }
        """
        bedrock_tools = []

        for tool in tools:
            # Support both "parameters" and "input_schema" keys
            schema = tool.get("parameters") or tool.get("input_schema", {})

            bedrock_tools.append({
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": {"json": schema}
                }
            })

        return bedrock_tools

    def _convert_response_to_standard(
        self,
        response: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert Bedrock response to standardized format.
        """
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        usage = response.get("usage", {})

        # Extract text content
        text_content = ""
        for block in content_blocks:
            if "text" in block:
                text_content += block["text"]

        return {
            "content": text_content,
            "model": self.model_id,
            "usage": {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
            },
            "finish_reason": self._map_stop_reason(response.get("stopReason", "unknown")),
            "raw_response": response,
        }

    def _convert_response_to_standard_with_tools(
        self,
        response: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert Bedrock response with potential tool calls to standardized format.
        """
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        usage = response.get("usage", {})

        # Extract text content and tool calls
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if "text" in block:
                text_content += block["text"]
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append({
                    "id": tool_use.get("toolUseId"),
                    "name": tool_use.get("name"),
                    "input": tool_use.get("input", {}),
                })

        return {
            "content": text_content if text_content else None,
            "tool_calls": tool_calls,
            "finish_reason": self._map_stop_reason(response.get("stopReason", "unknown")),
            "model": self.model_id,
            "usage": {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
            },
            "raw_response": response,
        }

    def _map_stop_reason(self, bedrock_reason: str) -> str:
        """Map Bedrock stop reasons to standardized values."""
        mapping = {
            "end_turn": "stop",
            "tool_use": "tool_use",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        return mapping.get(bedrock_reason, bedrock_reason)

    def _get_single_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> list[float]:
        """Get embedding for a single text using Titan."""
        request_body = {"inputText": text}

        if "v2" in self.embedding_model_id:
            request_body["normalize"] = normalize

        response = self.client.invoke_model(
            modelId=self.embedding_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )

        response_body = json.loads(response["body"].read())
        embedding = response_body["embedding"]

        # Normalize if needed and not done by model
        if normalize and "v2" not in self.embedding_model_id:
            embedding = self._normalize_vector(embedding)

        return embedding

    @staticmethod
    def _normalize_vector(vector: list[float]) -> list[float]:
        """Normalize a vector to unit length."""
        import math
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return vector
        return [x / magnitude for x in vector]

    # =========================================================================
    # Legacy Methods (for backward compatibility)
    # =========================================================================

    def invoke(
        self,
        prompt: str,
        messages: Optional[list[dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Legacy method - Send a message to Claude and get a response.

        This method is maintained for backward compatibility with existing code.
        For new code, prefer using generate() or generate_with_tools().

        Args:
            prompt: The user's message to send
            messages: Optional list of message history
            system_prompt: Optional system prompt
            tools: Optional list of tools
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Dictionary in legacy format with content (list), stop_reason, usage
        """
        logger.debug("Using legacy invoke() method")

        # Build messages list
        if messages is None:
            messages = []
        else:
            messages = list(messages)  # Copy to avoid modifying original

        if prompt:
            messages.append({
                "role": "user",
                "content": [{"text": prompt}]
            })

        # Convert to Bedrock format (messages are already in Bedrock format here)
        request_params = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            }
        }

        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]

        if tools:
            request_params["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "inputSchema": {
                                "json": tool.get("input_schema") or tool.get("parameters", {})
                            }
                        }
                    }
                    for tool in tools
                ]
            }

        try:
            response = self.client.converse(**request_params)

            output = response.get("output", {})
            message = output.get("message", {})

            result = {
                "content": message.get("content", []),
                "stop_reason": response.get("stopReason", "unknown"),
                "usage": response.get("usage", {}),
                "role": message.get("role", "assistant"),
            }

            usage = result["usage"]
            logger.info(
                f"Token usage - Input: {usage.get('inputTokens', 0)}, "
                f"Output: {usage.get('outputTokens', 0)}"
            )

            return result

        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise

    def invoke_with_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_use_id: str,
        tool_result: Any,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Legacy method - Continue a conversation after executing a tool.

        Maintained for backward compatibility.
        """
        tool_result_message = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": str(tool_result)}]
                    }
                }
            ]
        }

        updated_messages = messages + [tool_result_message]

        return self.invoke(
            prompt="",
            messages=updated_messages,
            system_prompt=system_prompt,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def stream_invoke(
        self,
        prompt: str,
        messages: Optional[list[dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Legacy method - Stream a response from Claude.

        Maintained for backward compatibility. For new code, use stream().
        """
        if messages is None:
            messages = []

        if prompt:
            messages.append({
                "role": "user",
                "content": [{"text": prompt}]
            })

        request_params = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            }
        }

        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]

        try:
            response = self.client.converse_stream(**request_params)

            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        yield delta["text"]
                elif "messageStart" in event:
                    logger.debug("Stream started")
                elif "messageStop" in event:
                    logger.debug(f"Stream ended: {event['messageStop']}")
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    logger.info(
                        f"Stream token usage - Input: {usage.get('inputTokens', 0)}, "
                        f"Output: {usage.get('outputTokens', 0)}"
                    )

        except ClientError as e:
            logger.error(f"Bedrock streaming error: {e}")
            raise


# =========================================================================
# Legacy Helper Functions (for backward compatibility)
# =========================================================================

def extract_text_from_response(response: dict[str, Any]) -> str:
    """
    Extract text content from a response.

    Works with both legacy format (content is list) and new format (content is str).
    """
    content = response.get("content", [])

    # Handle new standardized format (string)
    if isinstance(content, str):
        return content

    # Handle legacy format (list of content blocks)
    text_parts = []
    for content_block in content:
        if isinstance(content_block, dict) and "text" in content_block:
            text_parts.append(content_block["text"])
        elif isinstance(content_block, str):
            text_parts.append(content_block)

    return "\n".join(text_parts)


def extract_tool_use_from_response(response: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Extract tool use request from a response.

    Works with both legacy format and new format.
    """
    # Check for new standardized format
    tool_calls = response.get("tool_calls", [])
    if tool_calls:
        return tool_calls[0]  # Return first tool call for compatibility

    # Handle legacy format
    content = response.get("content", [])
    if isinstance(content, list):
        for content_block in content:
            if isinstance(content_block, dict) and "toolUse" in content_block:
                tool_use = content_block["toolUse"]
                return {
                    "id": tool_use.get("toolUseId"),
                    "name": tool_use.get("name"),
                    "input": tool_use.get("input", {}),
                }

    return None
