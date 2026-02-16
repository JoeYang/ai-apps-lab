"""
Session 2, Task 2: Tool Use (Function Calling)
================================================
Tool use is how you give LLMs the ability to interact with the real world.

The flow:
  1. You define tools (name, description, parameters)
  2. You send a user message + tool definitions to the model
  3. The model decides IF and WHICH tool to call, and with what arguments
  4. You execute the tool locally and return the result
  5. The model uses the result to form its final answer

KEY INSIGHT: The model never executes anything — it only generates
a structured request. YOU execute it. This is why it's safe.

Run: python 02_tool_use.py
"""

import json
import math
import os
from datetime import datetime

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5-20250929"


# ============================================================
# Step 1: Define your tools
# ============================================================
# Tools are described using JSON Schema — the model reads these
# descriptions to understand what each tool does and when to use it.

TOOLS = [
    {
        "name": "calculate",
        "description": "Perform mathematical calculations. Use this for any arithmetic, percentages, or mathematical operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g., '(100 * 0.23) / 4.5' or 'sqrt(144)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_system_metrics",
        "description": "Get current metrics for a trading system component. Returns CPU, memory, latency, and throughput data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "system_name": {
                    "type": "string",
                    "description": "Name of the system, e.g., 'matching-engine', 'fix-gateway', 'market-data-feed', 'risk-engine'"
                },
                "metric_type": {
                    "type": "string",
                    "enum": ["cpu", "memory", "latency", "throughput", "all"],
                    "description": "Type of metric to retrieve"
                }
            },
            "required": ["system_name", "metric_type"]
        }
    },
    {
        "name": "search_logs",
        "description": "Search recent log entries for a trading system component. Returns matching log lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "system_name": {
                    "type": "string",
                    "description": "Name of the system to search logs for"
                },
                "severity": {
                    "type": "string",
                    "enum": ["ERROR", "WARN", "INFO", "DEBUG"],
                    "description": "Minimum severity level to return"
                },
                "time_range_minutes": {
                    "type": "integer",
                    "description": "How many minutes back to search",
                    "default": 30
                }
            },
            "required": ["system_name", "severity"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time. Use when the user asks about current time or when you need timestamps.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


# ============================================================
# Step 2: Implement the actual tool functions
# ============================================================
# These are YOUR functions — they run locally, not on the LLM.
# In production, these would call real APIs, databases, etc.

def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    # In production, use a proper math parser — this is simplified
    allowed = set("0123456789+-*/().%^ sqrtabcdefghijlmnopqrtuvwxyz")
    if not all(c in allowed for c in expression.lower()):
        return f"Error: invalid characters in expression"
    try:
        # Replace common math functions
        expr = expression.replace("sqrt", "math.sqrt").replace("^", "**")
        result = eval(expr, {"__builtins__": {}, "math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_system_metrics(system_name: str, metric_type: str) -> str:
    """Simulated system metrics — in production this would query Datadog/Grafana/etc."""
    metrics = {
        "matching-engine": {
            "cpu": {"current": 67, "avg_1h": 52, "peak_1h": 89, "unit": "%"},
            "memory": {"used_gb": 28.4, "total_gb": 64, "unit": "GB"},
            "latency": {"p50_us": 12, "p99_us": 45, "p999_us": 120, "unit": "microseconds"},
            "throughput": {"orders_per_sec": 45000, "peak_1h": 78000, "unit": "orders/sec"},
        },
        "fix-gateway": {
            "cpu": {"current": 34, "avg_1h": 28, "peak_1h": 55, "unit": "%"},
            "memory": {"used_gb": 8.2, "total_gb": 32, "unit": "GB"},
            "latency": {"p50_us": 85, "p99_us": 340, "p999_us": 1200, "unit": "microseconds"},
            "throughput": {"messages_per_sec": 12000, "peak_1h": 25000, "unit": "msgs/sec"},
        },
        "market-data-feed": {
            "cpu": {"current": 78, "avg_1h": 65, "peak_1h": 95, "unit": "%"},
            "memory": {"used_gb": 45.1, "total_gb": 128, "unit": "GB"},
            "latency": {"p50_us": 5, "p99_us": 22, "p999_us": 88, "unit": "microseconds"},
            "throughput": {"updates_per_sec": 2500000, "peak_1h": 4200000, "unit": "updates/sec"},
        },
        "risk-engine": {
            "cpu": {"current": 45, "avg_1h": 38, "peak_1h": 72, "unit": "%"},
            "memory": {"used_gb": 16.8, "total_gb": 64, "unit": "GB"},
            "latency": {"p50_us": 150, "p99_us": 800, "p999_us": 3500, "unit": "microseconds"},
            "throughput": {"checks_per_sec": 30000, "peak_1h": 55000, "unit": "checks/sec"},
        },
    }

    if system_name not in metrics:
        return json.dumps({"error": f"Unknown system: {system_name}. Available: {list(metrics.keys())}"})

    if metric_type == "all":
        return json.dumps(metrics[system_name], indent=2)
    elif metric_type in metrics[system_name]:
        return json.dumps(metrics[system_name][metric_type], indent=2)
    else:
        return json.dumps({"error": f"Unknown metric: {metric_type}"})


def search_logs(system_name: str, severity: str, time_range_minutes: int = 30) -> str:
    """Simulated log search — in production this would query Splunk/ELK/etc."""
    logs = {
        "matching-engine": [
            {"time": "09:29:58.123", "severity": "WARN", "msg": "Order book depth below threshold for TSLA: 3 levels (min: 5)"},
            {"time": "09:30:01.456", "severity": "ERROR", "msg": "Reject: order 12345 price 185.50 outside circuit breaker band [180.00-190.00] for AAPL"},
            {"time": "09:30:02.789", "severity": "ERROR", "msg": "Reject: duplicate client order ID 'HFUND-98765' from session HEDGE_FUND_A"},
            {"time": "09:30:05.012", "severity": "WARN", "msg": "Matching latency spike: p99=320us (threshold: 100us) during NVDA auction"},
        ],
        "fix-gateway": [
            {"time": "09:28:12.345", "severity": "WARN", "msg": "Sequence gap detected: expected 45678, received 45680 from HEDGE_FUND_A"},
            {"time": "09:29:45.678", "severity": "ERROR", "msg": "Session HEDGE_FUND_A: 142 rejects in last 60s (threshold: 30)"},
            {"time": "09:30:01.234", "severity": "INFO", "msg": "Throttling HEDGE_FUND_A: rate reduced to 50 orders/sec"},
        ],
        "risk-engine": [
            {"time": "09:29:30.111", "severity": "WARN", "msg": "HEDGE_FUND_A margin utilisation at 94% (warning threshold: 90%)"},
            {"time": "09:29:55.222", "severity": "ERROR", "msg": "HEDGE_FUND_A buying power insufficient: required $2.3M, available $890K"},
            {"time": "09:30:00.333", "severity": "ERROR", "msg": "Position limit approaching: HEDGE_FUND_A TSLA position 48,500/50,000 (97%)"},
        ],
    }

    if system_name not in logs:
        return json.dumps({"error": f"Unknown system: {system_name}. Available: {list(logs.keys())}"})

    severity_order = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
    min_level = severity_order.get(severity, 0)
    filtered = [l for l in logs.get(system_name, []) if severity_order.get(l["severity"], 0) >= min_level]

    return json.dumps(filtered, indent=2)


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Map tool names to functions
TOOL_FUNCTIONS = {
    "calculate": lambda args: calculate(args["expression"]),
    "get_system_metrics": lambda args: get_system_metrics(args["system_name"], args["metric_type"]),
    "search_logs": lambda args: search_logs(args["system_name"], args["severity"], args.get("time_range_minutes", 30)),
    "get_current_time": lambda args: get_current_time(),
}


# ============================================================
# Step 3: The agent loop — this is the core pattern
# ============================================================
def run_agent(user_message: str):
    """
    The tool use loop:
      1. Send user message + tool definitions to the model
      2. If the model wants to use a tool → execute it, return result
      3. Repeat until the model gives a final text response

    The model may call MULTIPLE tools before giving a final answer.
    """
    print(f"\nUser: {user_message}")
    print("-" * 40)

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="""You are a trading systems support assistant. You have access to tools
for checking system metrics, searching logs, and performing calculations.
Always use the available tools to gather real data before answering.
Don't make up metrics — use the tools to get actual values.""",
            tools=TOOLS,
            messages=messages,
        )

        # Check what the model wants to do
        if response.stop_reason == "tool_use":
            # The model wants to call one or more tools
            # Collect all content blocks (text + tool calls)
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Process each tool call
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    print(f"  Tool call: {tool_name}({json.dumps(tool_input)})")

                    # Execute the tool
                    if tool_name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[tool_name](tool_input)
                    else:
                        result = f"Error: unknown tool {tool_name}"

                    print(f"  Result: {result[:200]}{'...' if len(result) > 200 else ''}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                    })

            # Send tool results back to the model
            messages.append({"role": "user", "content": tool_results})

        else:
            # The model is done — it has a final text response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            print(f"\nAssistant: {final_text}")
            return final_text


# ============================================================
# Run example queries
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SESSION 2: Tool Use (Function Calling)")
    print("=" * 60)

    # Example 1: Simple tool use — the model calls one tool
    run_agent("What's the current CPU usage on the matching engine?")

    print("\n" + "=" * 60)

    # Example 2: Multi-tool use — the model chains several tools
    run_agent(
        "HEDGE_FUND_A is complaining about order rejections. "
        "Check the FIX gateway logs, risk engine logs, and matching engine metrics "
        "to give me a full picture of what's happening."
    )

    print("\n" + "=" * 60)

    # Example 3: Tool use + calculation
    run_agent(
        "What percentage of the matching engine's memory is being used? "
        "And how does that compare to the risk engine?"
    )

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. You DEFINE tools — the model CHOOSES when to call them")
    print("  2. The model never executes anything — you run the tool locally")
    print("  3. Good tool descriptions are critical — they're like API docs for the model")
    print("  4. The model can chain multiple tools to answer complex questions")
    print("  5. This is the foundation of AI agents (Session 9+)")
    print("=" * 60)
