"""
Session 9, Task 2: Build a ReAct agent from scratch.

A minimal Reason → Act → Observe loop using the Anthropic SDK.
No frameworks — just the raw agent loop.
"""

import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
MODEL = "claude-sonnet-4-20250514"

# ── 1. Define tools (schema for Claude) ──────────────────────

tools = [
    {
        "name": "calculator",
        "description": "Evaluates a mathematical expression. Use this for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2 + 3 * 4'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Gets the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'London'",
                }
            },
            "required": ["city"],
        },
    },
]

# ── 2. Tool implementations ──────────────────────────────────


def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression)  # safe due to character whitelist
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Fake weather lookup (replace with real API later)."""
    weather_data = {
        "london": "Cloudy, 12°C, 80% humidity",
        "tokyo": "Sunny, 22°C, 45% humidity",
        "new york": "Rainy, 8°C, 90% humidity",
        "sydney": "Partly cloudy, 26°C, 55% humidity",
        "paris": "Overcast, 10°C, 75% humidity",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


# Map tool names to functions
tool_functions = {
    "calculator": lambda inputs: calculator(inputs["expression"]),
    "get_weather": lambda inputs: get_weather(inputs["city"]),
}

# ── 3. The Agent Loop ─────────────────────────────────────────


def run_agent(user_message: str, max_turns: int = 10) -> str:
    """Run the ReAct agent loop."""
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        print(f"\n── Turn {turn + 1} ──")

        # Call Claude
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful assistant. Think step-by-step before using tools.",
            tools=tools,
            messages=messages,
        )

        # Process the response content blocks
        assistant_content = response.content
        tool_results = []

        for block in assistant_content:
            if block.type == "text":
                print(f"THOUGHT: {block.text}")
            elif block.type == "tool_use":
                print(f"ACTION:  {block.name}({json.dumps(block.input)})")

                # Execute the tool
                if block.name in tool_functions:
                    result = tool_functions[block.name](block.input)
                else:
                    result = f"Error: unknown tool '{block.name}'"

                print(f"OBSERVE: {result}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": assistant_content})

        # If Claude used tools, feed results back and continue the loop
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            continue

        # If no tools were called, Claude is done (ANSWER)
        if response.stop_reason == "end_turn":
            final = next(
                (b.text for b in assistant_content if b.type == "text"), ""
            )
            print(f"\nANSWER: {final}")
            return final

    print("\nMax turns reached!")
    return "Agent stopped: too many turns."


# ── Run it ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: Simple tool call
    run_agent("What's 127 * 389?")

    # Test 2: Multi-step reasoning (tool + calculation)
    run_agent(
        "What's the weather in London, and what's the temperature in Fahrenheit?"
    )

    # Test 3: No tools needed — direct answer
    run_agent("What color is the sky?")
