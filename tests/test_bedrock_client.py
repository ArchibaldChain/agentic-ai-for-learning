"""
Tests for the Bedrock LLM client.

Run tests with:
    pytest tests/test_bedrock_client.py -v

These tests use mocking to avoid requiring AWS credentials.
Integration tests that require credentials are marked with @pytest.mark.skip.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import json

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_ai.llm.bedrock_client import (
    BedrockClient,
    extract_text_from_response,
    extract_tool_use_from_response,
)
from agentic_ai.llm.llm_client import (
    LLMClient,
    extract_text_content,
    extract_tool_calls,
    has_tool_calls,
)


class TestBedrockClientInitialization:
    """Tests for BedrockClient initialization."""

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_client_initialization(self, mock_get_config, mock_boto3):
        """Test that client initializes with correct config."""
        # Set up mock config
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        mock_config.embedding_model_id = "amazon.titan-embed-text-v1"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        client = BedrockClient()

        assert client.model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert client.embedding_model_id == "amazon.titan-embed-text-v1"
        mock_boto3.client.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-east-1"
        )

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_client_with_explicit_credentials(self, mock_get_config, mock_boto3):
        """Test client initialization with explicit credentials."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-west-2"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "test-embedding"
        mock_config.aws_access_key_id = "test_key_id"
        mock_config.aws_secret_access_key = "test_secret"
        mock_get_config.return_value = mock_config

        client = BedrockClient()

        mock_boto3.client.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-west-2",
            aws_access_key_id="test_key_id",
            aws_secret_access_key="test_secret"
        )

    def test_client_inherits_from_llm_client(self):
        """Test that BedrockClient inherits from LLMClient."""
        assert issubclass(BedrockClient, LLMClient)


class TestBedrockClientGenerate:
    """Tests for the generate() method."""

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_generate_simple_message(self, mock_get_config, mock_boto3):
        """Test basic message generation."""
        # Set up mocks
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "test-embedding"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Mock Bedrock response
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello! How can I help you?"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 8}
        }

        client = BedrockClient()
        response = client.generate([
            {"role": "user", "content": "Hello!"}
        ])

        # Check standardized response format
        assert response["content"] == "Hello! How can I help you?"
        assert response["model"] == "test-model"
        assert response["finish_reason"] == "stop"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 8
        assert response["usage"]["total_tokens"] == 18
        assert "raw_response" in response

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_generate_with_system_prompt(self, mock_get_config, mock_boto3):
        """Test generation with system prompt."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "test-embedding"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "I am a helpful assistant."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 20, "outputTokens": 10}
        }

        client = BedrockClient()
        client.generate(
            [{"role": "user", "content": "Who are you?"}],
            system_prompt="You are a helpful assistant."
        )

        # Verify system prompt was passed
        call_args = mock_client.converse.call_args
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"][0]["text"] == "You are a helpful assistant."


class TestBedrockClientGenerateWithTools:
    """Tests for the generate_with_tools() method."""

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_generate_with_tools_no_tool_call(self, mock_get_config, mock_boto3):
        """Test generation with tools available but not used."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "test-embedding"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "I don't need a tool for that."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 15, "outputTokens": 10}
        }

        client = BedrockClient()
        tools = [{
            "name": "calculator",
            "description": "A calculator",
            "parameters": {"type": "object", "properties": {}}
        }]

        response = client.generate_with_tools(
            [{"role": "user", "content": "What is your name?"}],
            tools=tools
        )

        assert response["content"] == "I don't need a tool for that."
        assert response["tool_calls"] == []
        assert response["finish_reason"] == "stop"

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_generate_with_tools_tool_call(self, mock_get_config, mock_boto3):
        """Test generation that results in a tool call."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "test-embedding"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Mock response with tool use
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Let me calculate that."},
                        {
                            "toolUse": {
                                "toolUseId": "tool_123",
                                "name": "calculator",
                                "input": {"expression": "2 + 2"}
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 20, "outputTokens": 15}
        }

        client = BedrockClient()
        tools = [{
            "name": "calculator",
            "description": "A calculator",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}}
            }
        }]

        response = client.generate_with_tools(
            [{"role": "user", "content": "What is 2 + 2?"}],
            tools=tools
        )

        assert response["content"] == "Let me calculate that."
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["id"] == "tool_123"
        assert response["tool_calls"][0]["name"] == "calculator"
        assert response["tool_calls"][0]["input"] == {"expression": "2 + 2"}
        assert response["finish_reason"] == "tool_use"


class TestBedrockClientGetEmbeddings:
    """Tests for the get_embeddings() method."""

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_get_embeddings_single(self, mock_get_config, mock_boto3):
        """Test getting embeddings for a single text."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "amazon.titan-embed-text-v1"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Mock embedding response
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = json.dumps({
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }).encode()
        mock_client.invoke_model.return_value = {"body": mock_response_body}

        client = BedrockClient()
        embeddings = client.get_embeddings(["Hello world"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 5
        # Check normalization was applied
        assert isinstance(embeddings[0][0], float)

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_get_embeddings_batch(self, mock_get_config, mock_boto3):
        """Test getting embeddings for multiple texts."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "amazon.titan-embed-text-v1"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Mock multiple embedding responses
        def create_response():
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({
                "embedding": [0.1, 0.2, 0.3]
            }).encode()
            return {"body": mock_body}

        mock_client.invoke_model.side_effect = [create_response() for _ in range(3)]

        client = BedrockClient()
        embeddings = client.get_embeddings(["Text 1", "Text 2", "Text 3"])

        assert len(embeddings) == 3
        assert mock_client.invoke_model.call_count == 3


class TestBedrockClientLegacyMethods:
    """Tests for legacy methods (backward compatibility)."""

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_invoke_legacy_method(self, mock_get_config, mock_boto3):
        """Test the legacy invoke() method still works."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "test-embedding"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Hello!"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3}
        }

        client = BedrockClient()
        response = client.invoke("Hi there!")

        # Legacy format returns content as list
        assert response["content"] == [{"text": "Hello!"}]
        assert response["stop_reason"] == "end_turn"
        assert response["role"] == "assistant"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_text_from_response_legacy_format(self):
        """Test extracting text from legacy format response."""
        response = {
            "content": [
                {"text": "Hello"},
                {"text": " World"}
            ]
        }
        text = extract_text_from_response(response)
        assert text == "Hello\n World"

    def test_extract_text_from_response_new_format(self):
        """Test extracting text from new standardized format."""
        response = {"content": "Hello World"}
        text = extract_text_from_response(response)
        assert text == "Hello World"

    def test_extract_text_from_response_empty(self):
        """Test extracting text from empty response."""
        response = {"content": []}
        text = extract_text_from_response(response)
        assert text == ""

    def test_extract_tool_use_from_response_with_tool(self):
        """Test extracting tool use from response with tool call."""
        response = {
            "content": [
                {"text": "Let me calculate."},
                {
                    "toolUse": {
                        "toolUseId": "tool_456",
                        "name": "calculator",
                        "input": {"expr": "1+1"}
                    }
                }
            ]
        }
        tool_use = extract_tool_use_from_response(response)
        assert tool_use is not None
        assert tool_use["id"] == "tool_456"
        assert tool_use["name"] == "calculator"
        assert tool_use["input"] == {"expr": "1+1"}

    def test_extract_tool_use_from_response_no_tool(self):
        """Test extracting tool use when there is none."""
        response = {"content": [{"text": "No tool needed."}]}
        tool_use = extract_tool_use_from_response(response)
        assert tool_use is None

    def test_extract_tool_use_from_new_format(self):
        """Test extracting tool use from new standardized format."""
        response = {
            "content": "Let me calculate.",
            "tool_calls": [
                {"id": "t1", "name": "calc", "input": {"x": 1}}
            ]
        }
        tool_use = extract_tool_use_from_response(response)
        assert tool_use["id"] == "t1"
        assert tool_use["name"] == "calc"


class TestLLMClientHelpers:
    """Tests for LLMClient helper functions."""

    def test_extract_text_content(self):
        """Test extract_text_content function."""
        response = {"content": "Hello World"}
        assert extract_text_content(response) == "Hello World"

    def test_extract_text_content_none(self):
        """Test extract_text_content with None content."""
        response = {"content": None}
        assert extract_text_content(response) == ""

    def test_extract_tool_calls(self):
        """Test extract_tool_calls function."""
        response = {
            "tool_calls": [
                {"id": "1", "name": "tool1", "input": {}},
                {"id": "2", "name": "tool2", "input": {}}
            ]
        }
        calls = extract_tool_calls(response)
        assert len(calls) == 2

    def test_extract_tool_calls_empty(self):
        """Test extract_tool_calls with no tools."""
        response = {"content": "No tools"}
        calls = extract_tool_calls(response)
        assert calls == []

    def test_has_tool_calls_true(self):
        """Test has_tool_calls returns True when tools present."""
        response = {"tool_calls": [{"id": "1", "name": "tool", "input": {}}]}
        assert has_tool_calls(response) is True

    def test_has_tool_calls_false(self):
        """Test has_tool_calls returns False when no tools."""
        response = {"tool_calls": []}
        assert has_tool_calls(response) is False


class TestMessageConversion:
    """Tests for message format conversion."""

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_convert_simple_messages(self, mock_get_config, mock_boto3):
        """Test converting simple messages to Bedrock format."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "test-embedding"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_boto3.client.return_value = MagicMock()

        client = BedrockClient()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        bedrock_messages = client._convert_messages_to_bedrock(messages)

        assert len(bedrock_messages) == 3
        assert bedrock_messages[0]["role"] == "user"
        assert bedrock_messages[0]["content"] == [{"text": "Hello"}]
        assert bedrock_messages[1]["role"] == "assistant"
        assert bedrock_messages[1]["content"] == [{"text": "Hi there!"}]

    @patch('agentic_ai.llm.bedrock_client.boto3')
    @patch('agentic_ai.llm.bedrock_client.get_config')
    def test_convert_tool_result_message(self, mock_get_config, mock_boto3):
        """Test converting tool result message to Bedrock format."""
        mock_config = MagicMock()
        mock_config.aws_region = "us-east-1"
        mock_config.bedrock_model_id = "test-model"
        mock_config.embedding_model_id = "test-embedding"
        mock_config.aws_access_key_id = None
        mock_config.aws_secret_access_key = None
        mock_get_config.return_value = mock_config

        mock_boto3.client.return_value = MagicMock()

        client = BedrockClient()
        messages = [
            {"role": "tool", "tool_call_id": "tool_123", "content": "Result: 4"}
        ]

        bedrock_messages = client._convert_messages_to_bedrock(messages)

        assert len(bedrock_messages) == 1
        assert bedrock_messages[0]["role"] == "user"
        assert "toolResult" in bedrock_messages[0]["content"][0]
        assert bedrock_messages[0]["content"][0]["toolResult"]["toolUseId"] == "tool_123"


class TestBedrockClientIntegration:
    """Integration tests that require AWS credentials."""

    @pytest.mark.skip(reason="Requires AWS credentials")
    def test_real_generation(self):
        """Test actual generation with AWS Bedrock."""
        client = BedrockClient()
        response = client.generate([
            {"role": "user", "content": "Say 'Hello' and nothing else."}
        ])
        assert "Hello" in response["content"]

    @pytest.mark.skip(reason="Requires AWS credentials")
    def test_real_embeddings(self):
        """Test actual embedding generation with AWS Bedrock."""
        client = BedrockClient()
        embeddings = client.get_embeddings(["Hello world"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0  # Titan returns 1536 dimensions
