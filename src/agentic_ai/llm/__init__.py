"""
LLM client modules for the agentic AI system.

This package provides a unified interface for interacting with different
LLM providers through the abstract LLMClient base class.

Available clients:
- BedrockClient: AWS Bedrock (Claude) - fully implemented
- AzureClient: Azure OpenAI - placeholder/stub

The architecture allows easy switching between providers:
    # Use Bedrock
    from agentic_ai.llm import BedrockClient
    client = BedrockClient()

    # Switch to Azure (when implemented)
    from agentic_ai.llm import AzureClient
    client = AzureClient()

    # Same interface for both!
    response = client.generate([{"role": "user", "content": "Hello"}])

All clients return standardized response formats, so agent code
doesn't need to change when switching providers.
"""

from .llm_client import (
    LLMClient,
    Message,
    ToolDefinition,
    extract_text_content,
    extract_tool_calls,
    has_tool_calls,
)
from .bedrock_client import (
    BedrockClient,
    extract_text_from_response,
    extract_tool_use_from_response,
)
from .azure_client import AzureClient

__all__ = [
    # Base class and utilities
    "LLMClient",
    "Message",
    "ToolDefinition",
    "extract_text_content",
    "extract_tool_calls",
    "has_tool_calls",
    # Bedrock implementation
    "BedrockClient",
    "extract_text_from_response",  # Legacy helper
    "extract_tool_use_from_response",  # Legacy helper
    # Azure implementation (placeholder)
    "AzureClient",
]
