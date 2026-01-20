"""
Base Agent implementing the ReAct (Reasoning + Acting) pattern.

The ReAct pattern is a simple but powerful approach to building AI agents:
1. The agent THINKS about what to do (reasoning)
2. The agent ACTS by calling a tool
3. The agent OBSERVES the result
4. Repeat until the task is complete

This module implements this loop from scratch using the Bedrock API,
showing exactly how agentic AI works under the hood.

Key concepts:
- Tools: Functions the agent can call (calculator, search, etc.)
- Agent Loop: The think-act-observe cycle
- Message History: Tracks the conversation for context
- Stop Conditions: When to end the loop (max iterations, task complete)

Example usage:
    from agentic_ai.agents.base_agent import BaseAgent, Tool

    # Define a simple calculator tool
    def calculator(expression: str) -> str:
        return str(eval(expression))

    calc_tool = Tool(
        name="calculator",
        description="Evaluates a mathematical expression",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g., '2 + 2'"
                }
            },
            "required": ["expression"]
        },
        function=calculator
    )

    agent = BaseAgent(tools=[calc_tool])
    result = agent.run("What is 15 multiplied by 7?")
    print(result)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..llm.bedrock_client import (
    BedrockClient,
    extract_text_from_response,
    extract_tool_use_from_response,
)
from ..utils.config import Config

# Set up logging to see the agent's thinking process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """
    Represents a tool that the agent can use.

    A tool is essentially a function with metadata that tells the LLM:
    - What the tool does (description)
    - What inputs it expects (input_schema)
    - How to call it (function)

    Attributes:
        name: Unique identifier for the tool (e.g., "calculator", "web_search")
        description: Human-readable description of what the tool does.
                     This is crucial - the LLM uses this to decide when to use the tool.
        input_schema: JSON Schema describing the expected input parameters.
                      This tells the LLM what arguments to provide.
        function: The actual Python function to call when the tool is invoked.
                  Should accept keyword arguments matching the input_schema.

    Example:
        >>> def search(query: str) -> str:
        ...     # Perform search
        ...     return "Search results..."
        >>> search_tool = Tool(
        ...     name="web_search",
        ...     description="Search the web for information",
        ...     input_schema={
        ...         "type": "object",
        ...         "properties": {
        ...             "query": {"type": "string", "description": "Search query"}
        ...         },
        ...         "required": ["query"]
        ...     },
        ...     function=search
        ... )
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[..., str]

    def to_bedrock_format(self) -> dict[str, Any]:
        """Convert to the format expected by Bedrock API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def execute(self, **kwargs) -> str:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Arguments matching the input_schema.

        Returns:
            String result from the tool.
        """
        logger.info(f"Executing tool '{self.name}' with args: {kwargs}")
        try:
            result = self.function(**kwargs)
            logger.info(f"Tool '{self.name}' returned: {result}")
            return str(result)
        except Exception as e:
            error_msg = f"Error executing tool '{self.name}': {str(e)}"
            logger.error(error_msg)
            return error_msg


class BaseAgent:
    """
    Base agent implementing the ReAct pattern.

    The ReAct (Reasoning + Acting) pattern works as follows:
    1. THINK: The LLM reasons about what to do next
    2. ACT: The LLM calls a tool if needed
    3. OBSERVE: The tool result is added to the conversation
    4. REPEAT: Continue until task is complete or max iterations reached

    The agent maintains a conversation history that includes:
    - The original user request
    - The agent's reasoning and tool calls
    - Tool results (observations)
    - The final answer

    Attributes:
        client: BedrockClient for LLM calls
        tools: List of available tools
        max_iterations: Maximum number of think-act-observe cycles
        system_prompt: Instructions that guide the agent's behavior
    """

    # Default system prompt that instructs the agent on how to behave
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that can use tools to accomplish tasks.

When given a task:
1. Think step-by-step about what you need to do
2. If you need information or need to perform an action, use the appropriate tool
3. After receiving tool results, analyze them and decide if you need more information
4. When you have enough information, provide a clear, helpful answer

Always explain your reasoning before using a tool. If you're unsure, say so.
Be concise but thorough in your final answers."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        config: Optional[Config] = None,
        max_iterations: int = 10,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            tools: List of Tool objects the agent can use. Default is empty list.
            config: Optional Config for the Bedrock client.
            max_iterations: Maximum think-act-observe cycles (default: 10).
                           This prevents infinite loops.
            system_prompt: Custom system prompt. If not provided, uses default.

        Example:
            >>> agent = BaseAgent(
            ...     tools=[calculator_tool, search_tool],
            ...     max_iterations=5
            ... )
        """
        self.client = BedrockClient(config=config)
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Create a lookup dict for quick tool access
        self._tool_map: dict[str, Tool] = {tool.name: tool for tool in self.tools}

        logger.info(
            f"Initialized BaseAgent with {len(self.tools)} tools: "
            f"{[t.name for t in self.tools]}"
        )

    def run(self, task: str, verbose: bool = True) -> str:
        """
        Run the agent on a task.

        This is the main entry point. It implements the ReAct loop:
        1. Send task to LLM with available tools
        2. If LLM wants to use a tool, execute it and continue
        3. If LLM is done (no tool use), return the final answer

        Args:
            task: The task or question for the agent to handle.
            verbose: If True, print the agent's thinking process (default: True).

        Returns:
            The agent's final response as a string.

        Example:
            >>> agent = BaseAgent(tools=[calculator_tool])
            >>> result = agent.run("What is 123 * 456?")
            >>> print(result)
            "123 * 456 = 56,088"
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TASK: {task}")
            print(f"{'='*60}\n")

        # Initialize message history with the user's task
        # The message history tracks the full conversation
        messages: list[dict[str, Any]] = []

        # Convert tools to Bedrock format
        bedrock_tools = [tool.to_bedrock_format() for tool in self.tools] if self.tools else None

        # Track iterations to prevent infinite loops
        iteration = 0

        # === THE REACT LOOP ===
        # This is the core of the agent - the think-act-observe cycle
        while iteration < self.max_iterations:
            iteration += 1

            if verbose:
                print(f"\n--- Iteration {iteration} ---")

            # STEP 1: THINK
            # Send the current state to the LLM and let it reason
            if iteration == 1:
                # First iteration: send the original task
                response = self.client.invoke(
                    prompt=task,
                    messages=messages,
                    system_prompt=self.system_prompt,
                    tools=bedrock_tools,
                )
            else:
                # Subsequent iterations: continue the conversation
                # (messages already includes tool results)
                response = self.client.invoke(
                    prompt="",  # No new prompt, just continuing
                    messages=messages,
                    system_prompt=self.system_prompt,
                    tools=bedrock_tools,
                )

            # Add the assistant's response to the message history
            # This is important for maintaining conversation context
            assistant_message = {
                "role": "assistant",
                "content": response["content"]
            }
            messages.append(assistant_message)

            # Extract text response (the agent's thinking)
            text_response = extract_text_from_response(response)
            if verbose and text_response:
                print(f"\nAGENT THINKING:\n{text_response}")

            # Check if the agent wants to use a tool
            tool_use = extract_tool_use_from_response(response)

            if tool_use:
                # STEP 2: ACT
                # The agent wants to use a tool - execute it
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]
                tool_use_id = tool_use["id"]

                if verbose:
                    print(f"\nTOOL CALL: {tool_name}")
                    print(f"INPUT: {tool_input}")

                # Find and execute the tool
                if tool_name in self._tool_map:
                    tool = self._tool_map[tool_name]
                    tool_result = tool.execute(**tool_input)
                else:
                    tool_result = f"Error: Unknown tool '{tool_name}'"

                # STEP 3: OBSERVE
                # Add the tool result to the conversation
                if verbose:
                    print(f"\nTOOL RESULT: {tool_result}")

                # Format tool result as a user message (this is how Bedrock expects it)
                tool_result_message = {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": tool_result}]
                            }
                        }
                    ]
                }
                messages.append(tool_result_message)

                # Continue the loop - the agent will process the tool result
            else:
                # No tool use - the agent is done!
                # This happens when stop_reason is "end_turn"
                if verbose:
                    print(f"\n{'='*60}")
                    print("FINAL ANSWER:")
                    print(f"{'='*60}")
                    print(text_response)

                return text_response

        # Reached max iterations without completing
        final_message = (
            f"Agent reached maximum iterations ({self.max_iterations}) "
            "without completing the task."
        )
        logger.warning(final_message)

        # Try to extract any partial response
        if messages:
            last_response = extract_text_from_response({"content": messages[-1].get("content", [])})
            if last_response:
                return last_response

        return final_message

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent.

        Args:
            tool: Tool object to add.
        """
        self.tools.append(tool)
        self._tool_map[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent.

        Args:
            tool_name: Name of the tool to remove.

        Returns:
            True if tool was removed, False if not found.
        """
        if tool_name in self._tool_map:
            del self._tool_map[tool_name]
            self.tools = [t for t in self.tools if t.name != tool_name]
            logger.info(f"Removed tool: {tool_name}")
            return True
        return False


# === EXAMPLE TOOLS ===
# These are simple example tools to demonstrate the pattern


def create_calculator_tool() -> Tool:
    """
    Create a simple calculator tool.

    Returns:
        Tool that can evaluate mathematical expressions.

    Example:
        >>> calc = create_calculator_tool()
        >>> agent = BaseAgent(tools=[calc])
        >>> agent.run("What is 2^10?")
    """
    def calculator(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        # Note: In production, use a proper math parser instead of eval
        # This is simplified for educational purposes
        try:
            # Only allow safe characters
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    return Tool(
        name="calculator",
        description=(
            "A calculator that evaluates mathematical expressions. "
            "Use this for any math calculations. "
            "Input should be a valid mathematical expression like '2 + 2' or '15 * 7'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g., '2 + 2', '15 * 7', '100 / 4'"
                }
            },
            "required": ["expression"]
        },
        function=calculator
    )


def create_current_time_tool() -> Tool:
    """
    Create a tool that returns the current time.

    Returns:
        Tool that returns the current date and time.
    """
    from datetime import datetime

    def get_current_time() -> str:
        """Return the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return Tool(
        name="current_time",
        description="Get the current date and time. Use this when asked about the current time or date.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        },
        function=get_current_time
    )
