"""
Session 10, Task 1: LangChain Basics.

Demonstrates the core LangChain abstractions:
- Chat models
- Prompt templates
- Chains (LCEL)
- Tools and tool-calling agents
- Callbacks for observability
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler

# ── 1. Chat Models ───────────────────────────────────────────
# LangChain wraps different LLM providers behind a common interface.
# Swap ChatAnthropic for ChatOpenAI and the rest of your code stays the same.

print("=" * 60)
print("1. CHAT MODELS")
print("=" * 60)

model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=256)

# Simple invocation — same interface regardless of provider
response = model.invoke([HumanMessage(content="What is ReAct in one sentence?")])
print(f"Response: {response.content}\n")

# ── 2. Prompt Templates ─────────────────────────────────────
# Reusable templates with variables. Way cleaner than f-strings.

print("=" * 60)
print("2. PROMPT TEMPLATES")
print("=" * 60)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on {topic}. Be concise."),
        ("human", "{question}"),
    ]
)

# Invoke the template to see what it produces
formatted = prompt.invoke({"topic": "AI agents", "question": "What is tool use?"})
print(f"Formatted messages: {formatted.to_messages()}\n")

# ── 3. Chains (LCEL — LangChain Expression Language) ─────────
# The | operator pipes components together: prompt → model → parser

print("=" * 60)
print("3. CHAINS (LCEL)")
print("=" * 60)

# This is a chain: prompt feeds into model, model output parsed to string
chain = prompt | model | StrOutputParser()

result = chain.invoke(
    {"topic": "AI agents", "question": "What are the 3 main failure modes?"}
)
print(f"Chain result:\n{result}\n")

# ── 4. Tools ─────────────────────────────────────────────────
# LangChain tools use the @tool decorator. The docstring becomes
# the tool description that the LLM sees.

print("=" * 60)
print("4. TOOLS")
print("=" * 60)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for any arithmetic."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def lookup_capital(country: str) -> str:
    """Look up the capital city of a country."""
    capitals = {
        "france": "Paris",
        "japan": "Tokyo",
        "australia": "Canberra",
        "brazil": "Brasilia",
    }
    return capitals.get(country.lower(), f"Unknown capital for {country}")


# Tools are just functions with metadata
print(f"Tool name: {calculator.name}")
print(f"Tool description: {calculator.description}")
print(f"Tool schema: {calculator.args_schema.model_json_schema()}\n")

# ── 5. Tool-calling with bind_tools ─────────────────────────
# Bind tools to the model — LangChain handles the schema conversion

print("=" * 60)
print("5. TOOL-CALLING AGENT (simple)")
print("=" * 60)

tools = [calculator, lookup_capital]
model_with_tools = model.bind_tools(tools)

# The model now returns tool_calls when it wants to use a tool
response = model_with_tools.invoke(
    [HumanMessage(content="What's the capital of France, and what's 127 * 389?")]
)

print(f"Response type: {type(response)}")
print(f"Content: {response.content}")
print(f"Tool calls: {response.tool_calls}\n")

# ── 6. Callbacks for Observability ───────────────────────────
# Callbacks let you hook into every step — great for logging/debugging.

print("=" * 60)
print("6. CALLBACKS")
print("=" * 60)


class LoggingCallback(BaseCallbackHandler):
    """Simple callback that logs LLM start/end events."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"  [CALLBACK] LLM started")

    def on_llm_end(self, response, **kwargs):
        print(f"  [CALLBACK] LLM finished")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"  [CALLBACK] Tool started: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"  [CALLBACK] Tool finished: {output}")


# Run a chain with the callback
chain_with_logging = prompt | model | StrOutputParser()
result = chain_with_logging.invoke(
    {"topic": "LangChain", "question": "What is LCEL in one sentence?"},
    config={"callbacks": [LoggingCallback()]},
)
print(f"Result: {result}\n")

# ── Summary ──────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY: LangChain Core Abstractions")
print("=" * 60)
print(
    """
ChatModel      - Uniform interface to any LLM (Claude, GPT-4, etc.)
PromptTemplate - Reusable, parameterised prompts
Chain (LCEL)   - Compose components with | operator (prompt | model | parser)
Tool           - @tool decorator turns functions into LLM-callable tools
bind_tools()   - Attach tools to a model so it can call them
Callbacks      - Hook into events for logging, tracing, debugging
"""
)
