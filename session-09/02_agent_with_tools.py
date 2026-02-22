"""
Session 9, Task 3: ReAct agent with powerful tools.

Extends the basic agent with: file I/O, code execution, and web search.
"""

import anthropic
import json
import os
import subprocess
import tempfile
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"

# ── Tool Definitions (schemas for Claude) ─────────────────────

tools = [
    {
        "name": "calculator",
        "description": "Evaluates a mathematical expression. Use for arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2 + 3 * 4'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to current directory.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_python",
        "description": "Execute Python code and return stdout/stderr. Use for computation, data processing, or any task requiring code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information. Returns search results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
            "required": ["query"],
        },
    },
]

# ── Tool Implementations ──────────────────────────────────────

# Sandbox: restrict file operations to a working directory
SANDBOX_DIR = os.path.join(os.path.dirname(__file__), "agent_workspace")
os.makedirs(SANDBOX_DIR, exist_ok=True)


def _safe_path(path: str) -> str:
    """Resolve path within the sandbox directory to prevent escapes."""
    resolved = os.path.realpath(os.path.join(SANDBOX_DIR, path))
    if not resolved.startswith(os.path.realpath(SANDBOX_DIR)):
        raise ValueError(f"Path escapes sandbox: {path}")
    return resolved


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    try:
        safe = _safe_path(path)
        with open(safe) as f:
            content = f.read()
        # Truncate very large files
        if len(content) > 10000:
            return content[:10000] + f"\n... (truncated, {len(content)} chars total)"
        return content
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    try:
        safe = _safe_path(path)
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        with open(safe, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_directory(path: str) -> str:
    try:
        safe = _safe_path(path)
        entries = os.listdir(safe)
        if not entries:
            return "(empty directory)"
        result = []
        for entry in sorted(entries):
            full = os.path.join(safe, entry)
            kind = "dir" if os.path.isdir(full) else "file"
            result.append(f"  {entry} ({kind})")
        return "\n".join(result)
    except FileNotFoundError:
        return f"Error: directory not found: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def run_python(code: str) -> str:
    """Run Python code in a subprocess with a timeout."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=SANDBOX_DIR,
        )
        os.unlink(tmp_path)

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        os.unlink(tmp_path)
        return "Error: code execution timed out (10s limit)"
    except Exception as e:
        return f"Error executing code: {e}"


def web_search(query: str) -> str:
    """Simulated web search. In production, use a real search API."""
    # Simulated results — replace with SerpAPI, Brave Search, etc.
    fake_results = {
        "python": "Python is a high-level programming language. Latest version: 3.13. Created by Guido van Rossum in 1991.",
        "react": "React is a JavaScript library for building UIs, maintained by Meta. Latest: React 19.",
        "anthropic": "Anthropic is an AI safety company. They build Claude, a family of AI assistants.",
        "langchain": "LangChain is an open-source framework for building LLM applications. Supports agents, RAG, and chains.",
    }
    # Simple keyword matching
    query_lower = query.lower()
    results = []
    for keyword, info in fake_results.items():
        if keyword in query_lower:
            results.append(info)
    if results:
        return "\n".join(results)
    return f"Search results for '{query}': No relevant results found. (This is a simulated search — connect a real API for production use.)"


# Tool dispatch
tool_functions = {
    "calculator": lambda inp: calculator(inp["expression"]),
    "read_file": lambda inp: read_file(inp["path"]),
    "write_file": lambda inp: write_file(inp["path"], inp["content"]),
    "list_directory": lambda inp: list_directory(inp["path"]),
    "run_python": lambda inp: run_python(inp["code"]),
    "web_search": lambda inp: web_search(inp["query"]),
}

# ── Agent Loop ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant with access to tools for file I/O, \
code execution, calculation, and web search.

Think step-by-step before acting. Use tools when you need real data or computation. \
File operations are sandboxed to the agent_workspace directory.

When writing code with run_python, always print() your results so they appear in the output."""


def run_agent(user_message: str, max_turns: int = 15) -> str:
    """Run the ReAct agent loop."""
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        print(f"\n── Turn {turn + 1} ──")

        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        assistant_content = response.content
        tool_results = []

        for block in assistant_content:
            if block.type == "text":
                print(f"THOUGHT: {block.text}")
            elif block.type == "tool_use":
                # Truncate long inputs for display
                input_str = json.dumps(block.input)
                display = input_str[:120] + "..." if len(input_str) > 120 else input_str
                print(f"ACTION:  {block.name}({display})")

                if block.name in tool_functions:
                    result = tool_functions[block.name](block.input)
                else:
                    result = f"Error: unknown tool '{block.name}'"

                # Truncate long output for display
                display_result = result[:200] + "..." if len(result) > 200 else result
                print(f"OBSERVE: {display_result}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        messages.append({"role": "assistant", "content": assistant_content})

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "end_turn":
            final = next(
                (b.text for b in assistant_content if b.type == "text"), ""
            )
            print(f"\nANSWER: {final}")
            return final

    print("\nMax turns reached!")
    return "Agent stopped: too many turns."


# ── Test Scenarios ────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: Code execution — agent writes and runs Python
    run_agent("What are the first 20 prime numbers? Use code to compute them.")

    # Test 2: File I/O — agent creates and reads back a file
    run_agent(
        "Create a file called 'notes.md' with a summary of the ReAct agent pattern, "
        "then read it back to confirm."
    )

    # Test 3: Multi-tool — agent uses multiple tools together
    run_agent(
        "List the files in the current directory, then write a Python script "
        "called 'hello.py' that prints 'Hello from the agent!', and run it."
    )
