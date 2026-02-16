"""
Session 2, Task 3: Interactive Tool-Using Assistant
====================================================
A conversational assistant that can:
  - Read files from your filesystem
  - Do calculations
  - Search the web (simulated)
  - Analyse trading data

This combines everything from Tasks 1 & 2:
  - Structured outputs (Pydantic) for the tools
  - Tool use loop for execution
  - Multi-turn conversation for context

Run: python 03_tool_assistant.py
Then type questions interactively. Type 'quit' to exit.

Try:
  > Read the file session-01/01_basic_api_calls.py and tell me how many functions it defines
  > What is 15% of 2.5 million?
  > Search for best practices for FIX session management
  > Analyse this order data: AAPL 1000@185, TSLA 500@220, NVDA 2000@450 — what's my total exposure?
"""

import json
import math
import os
import urllib.request
import urllib.parse

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5-20250929"

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from the local filesystem. Use this when the user asks about a file or wants you to analyse code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to current directory or absolute)"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files and directories at a given path. Use this to explore the filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                    "default": "."
                }
            },
            "required": []
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports basic arithmetic, sqrt, pow, percentages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g., '1000 * 185.50 + 500 * 220' or 'sqrt(144)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather information for a city. Returns temperature, wind speed, and weather conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g., 'London', 'New York', 'Tokyo'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for information. Returns a summary of top results. Use for current events, best practices, documentation lookups.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    },
]


# ============================================================
# Tool implementations
# ============================================================

def read_file(file_path: str) -> str:
    """Read a file from the filesystem."""
    try:
        # Resolve relative paths from the ai-apps-lab directory
        if not os.path.isabs(file_path):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base, file_path)

        with open(file_path, "r") as f:
            content = f.read()

        # Truncate very large files
        if len(content) > 10000:
            content = content[:10000] + f"\n\n... [truncated, file is {len(content)} chars total]"

        return content
    except FileNotFoundError:
        return f"Error: file not found: {file_path}"
    except PermissionError:
        return f"Error: permission denied: {file_path}"


def list_directory(path: str = ".") -> str:
    """List directory contents."""
    try:
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, path)

        entries = sorted(os.listdir(path))
        result = []
        for entry in entries:
            full = os.path.join(path, entry)
            entry_type = "dir" if os.path.isdir(full) else "file"
            size = os.path.getsize(full) if os.path.isfile(full) else ""
            result.append(f"  {entry_type:4s}  {entry}" + (f"  ({size} bytes)" if size else ""))

        return f"Contents of {path}:\n" + "\n".join(result)
    except FileNotFoundError:
        return f"Error: directory not found: {path}"


def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        allowed_names = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Get current weather for a city using the Open-Meteo API (free, no API key)."""
    try:
        # Step 1: Geocode the city name to lat/lon
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
            {"name": city, "count": 1}
        )
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo_data = json.loads(resp.read())

        if "results" not in geo_data or not geo_data["results"]:
            return f"Error: could not find city '{city}'"

        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        resolved_name = location.get("name", city)
        country = location.get("country", "")

        # Step 2: Fetch current weather
        weather_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        })
        with urllib.request.urlopen(weather_url, timeout=10) as resp:
            weather_data = json.loads(resp.read())

        current = weather_data["current"]

        # Map WMO weather codes to descriptions
        wmo_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
        }
        condition = wmo_codes.get(current.get("weather_code", -1), "Unknown")

        return json.dumps({
            "city": resolved_name,
            "country": country,
            "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
            "wind_speed_kmh": current["wind_speed_10m"],
            "condition": condition,
        }, indent=2)

    except Exception as e:
        return f"Error fetching weather: {e}"


def web_search(query: str) -> str:
    """Simulated web search — in production, use a real search API."""
    # This is simulated. In a real app, you'd call Google/Bing/Brave Search API.
    simulated_results = {
        "fix": [
            {"title": "FIX Protocol Best Practices 2025", "snippet": "Key practices: implement heartbeat monitoring with 30s intervals, use sequence number gap detection, maintain backup sessions, implement automatic reconnection with exponential backoff."},
            {"title": "FIX Session Management Guide", "snippet": "Always validate sequence numbers on logon. Implement ResendRequest handling. Use TestRequest for keepalive. Set appropriate MaxMessageSize limits."},
        ],
        "trading": [
            {"title": "Trading System Architecture Guide", "snippet": "Modern trading systems use event-driven architecture with message queues. Key components: order gateway, matching engine, risk engine, market data handler."},
            {"title": "Low-Latency Trading Best Practices", "snippet": "Kernel bypass (DPDK/RDMA), lock-free data structures, memory-mapped I/O, CPU pinning, NUMA-aware allocation."},
        ],
        "default": [
            {"title": "Search result for: " + query, "snippet": "This is a simulated search result. In production, connect to a real search API like Google Custom Search, Bing Web Search, or Brave Search API."},
        ],
    }

    # Match query to simulated results
    for keyword, results in simulated_results.items():
        if keyword in query.lower():
            return json.dumps(results, indent=2)

    return json.dumps(simulated_results["default"], indent=2)


TOOL_FUNCTIONS = {
    "read_file": lambda args: read_file(args["file_path"]),
    "list_directory": lambda args: list_directory(args.get("path", ".")),
    "calculate": lambda args: calculate(args["expression"]),
    "get_weather": lambda args: get_weather(args["city"]),
    "web_search": lambda args: web_search(args["query"]),
}


# ============================================================
# Interactive agent loop
# ============================================================

def run_interactive():
    """Multi-turn interactive assistant with tool use."""
    messages = []

    print("=" * 60)
    print("Trading Systems Assistant (with tools)")
    print("Tools: read_file, list_directory, calculate, get_weather, web_search")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Agent loop — keep going until the model gives a final text answer
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system="""You are a helpful trading systems assistant. You can read files,
list directories, do calculations, and search the web.
Always use tools when they would help answer the question accurately.
Be concise in your answers.""",
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  [tool: {block.name}({json.dumps(block.input)[:80]}...)]")

                        result = TOOL_FUNCTIONS[block.name](block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

            else:
                # Final answer
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                messages.append({"role": "assistant", "content": final_text})
                print(f"\n{final_text}")
                break


if __name__ == "__main__":
    run_interactive()
