"""
Agent Example.

This example demonstrates the ReAct agent pattern:
1. Create tools that the agent can use
2. Initialize the agent with those tools
3. Give the agent a task
4. Watch the agent think, act, and observe

The ReAct (Reasoning + Acting) pattern:
- THINK: The agent reasons about what to do
- ACT: The agent calls a tool
- OBSERVE: The agent sees the result
- REPEAT: Until the task is complete

Run this example with:
    python examples/agent_example.py

Make sure you have:
1. Set up your .env file with AWS credentials
2. Installed dependencies with: uv pip install -e .
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_ai.agents.base_agent import (
    BaseAgent,
    Tool,
    create_calculator_tool,
    create_current_time_tool,
)


def create_weather_tool() -> Tool:
    """
    Create a mock weather tool for demonstration.

    In a real application, this would call a weather API.
    """
    def get_weather(city: str) -> str:
        """Get weather for a city (mock implementation)."""
        # Mock weather data
        weather_data = {
            "new york": "Sunny, 72°F (22°C)",
            "london": "Cloudy, 59°F (15°C)",
            "tokyo": "Rainy, 68°F (20°C)",
            "paris": "Partly cloudy, 64°F (18°C)",
            "sydney": "Clear, 77°F (25°C)",
        }

        city_lower = city.lower()
        if city_lower in weather_data:
            return f"Weather in {city}: {weather_data[city_lower]}"
        else:
            return f"Weather in {city}: Sunny, 70°F (21°C) (default)"

    return Tool(
        name="get_weather",
        description=(
            "Get the current weather for a city. "
            "Use this when asked about weather conditions. "
            "Input should be a city name like 'New York' or 'London'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for"
                }
            },
            "required": ["city"]
        },
        function=get_weather,
    )


def create_unit_converter_tool() -> Tool:
    """Create a unit conversion tool."""
    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert between units."""
        conversions = {
            ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
            ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
            ("miles", "kilometers"): lambda x: x * 1.60934,
            ("kilometers", "miles"): lambda x: x / 1.60934,
            ("pounds", "kilograms"): lambda x: x * 0.453592,
            ("kilograms", "pounds"): lambda x: x / 0.453592,
            ("feet", "meters"): lambda x: x * 0.3048,
            ("meters", "feet"): lambda x: x / 0.3048,
        }

        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.2f} {to_unit}"
        else:
            return f"Cannot convert from {from_unit} to {to_unit}"

    return Tool(
        name="convert_units",
        description=(
            "Convert between different units of measurement. "
            "Supports: temperature (celsius/fahrenheit), "
            "distance (miles/kilometers, feet/meters), "
            "weight (pounds/kilograms)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The numeric value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "description": "The unit to convert from (e.g., 'celsius', 'miles')"
                },
                "to_unit": {
                    "type": "string",
                    "description": "The unit to convert to (e.g., 'fahrenheit', 'kilometers')"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        },
        function=convert_units,
    )


def run_simple_example():
    """Run a simple calculator example."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Calculator")
    print("=" * 60)

    # Create agent with calculator tool
    calc_tool = create_calculator_tool()
    agent = BaseAgent(tools=[calc_tool])

    # Run a math task
    task = "What is 15 multiplied by 7, plus 23?"
    result = agent.run(task, verbose=True)

    return result


def run_multi_tool_example():
    """Run an example with multiple tools."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Tools")
    print("=" * 60)

    # Create agent with multiple tools
    tools = [
        create_calculator_tool(),
        create_weather_tool(),
        create_current_time_tool(),
        create_unit_converter_tool(),
    ]

    agent = BaseAgent(tools=tools, max_iterations=5)

    # Run a task that requires multiple tools
    task = (
        "I'm planning a trip to London. Can you tell me: "
        "1) What's the current weather there? "
        "2) Convert the temperature to Fahrenheit if it's in Celsius."
    )

    result = agent.run(task, verbose=True)
    return result


def run_reasoning_example():
    """Run an example that shows the agent's reasoning."""
    print("\n" + "=" * 60)
    print("Example 3: Complex Reasoning")
    print("=" * 60)

    tools = [
        create_calculator_tool(),
        create_unit_converter_tool(),
    ]

    agent = BaseAgent(tools=tools, max_iterations=5)

    # A task that requires multiple steps
    task = (
        "If I drive 150 miles and my car gets 30 miles per gallon, "
        "how many liters of gas will I use? "
        "(Note: 1 gallon = 3.78541 liters)"
    )

    result = agent.run(task, verbose=True)
    return result


def main():
    """Run all agent examples."""
    print("=" * 60)
    print("Agent Example - ReAct Pattern Demonstration")
    print("=" * 60)
    print("\nThis example shows how an AI agent uses the ReAct pattern:")
    print("1. THINK: Reason about what to do")
    print("2. ACT: Call a tool")
    print("3. OBSERVE: See the result")
    print("4. REPEAT: Until task is complete")

    try:
        # Run examples
        run_simple_example()

        print("\n" + "-" * 60)
        input("Press Enter to continue to Example 2...")

        run_multi_tool_example()

        print("\n" + "-" * 60)
        input("Press Enter to continue to Example 3...")

        run_reasoning_example()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure your AWS credentials are configured correctly in .env")

    print("\n" + "=" * 60)
    print("Agent Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
