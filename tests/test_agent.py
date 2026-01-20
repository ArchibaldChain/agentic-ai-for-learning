"""
Tests for the base agent.

Run tests with:
    pytest tests/test_agent.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_ai.agents.base_agent import Tool, BaseAgent, create_calculator_tool


class TestTool:
    """Tests for the Tool class."""

    def test_tool_creation(self):
        """Test creating a tool."""
        def dummy_func(x: str) -> str:
            return f"Result: {x}"

        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            },
            function=dummy_func,
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.function == dummy_func

    def test_tool_execute(self):
        """Test executing a tool."""
        def add_numbers(a: int, b: int) -> str:
            return str(a + b)

        tool = Tool(
            name="add",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            },
            function=add_numbers,
        )

        result = tool.execute(a=5, b=3)
        assert result == "8"

    def test_tool_execute_error(self):
        """Test tool execution with error."""
        def failing_func() -> str:
            raise ValueError("Test error")

        tool = Tool(
            name="failing",
            description="A failing tool",
            input_schema={"type": "object", "properties": {}},
            function=failing_func,
        )

        result = tool.execute()
        assert "Error" in result
        assert "Test error" in result

    def test_tool_to_bedrock_format(self):
        """Test converting tool to Bedrock format."""
        tool = Tool(
            name="test",
            description="Test description",
            input_schema={"type": "object", "properties": {}},
            function=lambda: "test",
        )

        bedrock_format = tool.to_bedrock_format()

        assert bedrock_format["name"] == "test"
        assert bedrock_format["description"] == "Test description"
        assert "input_schema" in bedrock_format


class TestCalculatorTool:
    """Tests for the calculator tool."""

    def test_calculator_addition(self):
        """Test calculator addition."""
        calc = create_calculator_tool()
        result = calc.execute(expression="2 + 2")
        assert result == "4"

    def test_calculator_multiplication(self):
        """Test calculator multiplication."""
        calc = create_calculator_tool()
        result = calc.execute(expression="15 * 7")
        assert result == "105"

    def test_calculator_complex(self):
        """Test calculator with complex expression."""
        calc = create_calculator_tool()
        result = calc.execute(expression="(10 + 5) * 2")
        assert result == "30"

    def test_calculator_invalid_chars(self):
        """Test calculator rejects invalid characters."""
        calc = create_calculator_tool()
        result = calc.execute(expression="import os")
        assert "Error" in result or "Invalid" in result


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    def test_agent_creation(self):
        """Test creating an agent."""
        agent = BaseAgent(tools=[], max_iterations=5)
        assert agent.max_iterations == 5
        assert len(agent.tools) == 0

    def test_agent_add_tool(self):
        """Test adding a tool to agent."""
        agent = BaseAgent(tools=[])
        calc = create_calculator_tool()

        agent.add_tool(calc)

        assert len(agent.tools) == 1
        assert "calculator" in agent._tool_map

    def test_agent_remove_tool(self):
        """Test removing a tool from agent."""
        calc = create_calculator_tool()
        agent = BaseAgent(tools=[calc])

        result = agent.remove_tool("calculator")

        assert result is True
        assert len(agent.tools) == 0

    def test_agent_remove_nonexistent_tool(self):
        """Test removing a tool that doesn't exist."""
        agent = BaseAgent(tools=[])

        result = agent.remove_tool("nonexistent")

        assert result is False

    @patch('agentic_ai.agents.base_agent.BedrockClient')
    def test_agent_run_simple(self, mock_client_class):
        """Test agent run with mocked Bedrock client."""
        # Set up mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock a simple response (no tool use)
        mock_client.invoke.return_value = {
            "content": [{"text": "The answer is 42."}],
            "stop_reason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        agent = BaseAgent(tools=[])
        result = agent.run("What is the answer?", verbose=False)

        assert result == "The answer is 42."
        mock_client.invoke.assert_called_once()


class TestAgentIntegration:
    """Integration tests (require AWS credentials)."""

    @pytest.mark.skip(reason="Requires AWS credentials")
    def test_agent_with_calculator(self):
        """Test agent with calculator tool (integration test)."""
        calc = create_calculator_tool()
        agent = BaseAgent(tools=[calc])

        result = agent.run("What is 7 times 8?", verbose=True)

        assert "56" in result
