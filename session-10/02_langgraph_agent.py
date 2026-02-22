"""
Session 10, Task 2: Build a LangGraph agent.

Demonstrates the graph-based agent pattern:
- State definition (TypedDict with message list)
- Nodes (agent node, tool node)
- Conditional edges (should we call tools or stop?)
- The full agent loop as an explicit graph
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ── Tools ─────────────────────────────────────────────────────


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
def search_knowledge(query: str) -> str:
    """Search an internal knowledge base. Returns relevant information."""
    kb = {
        "vacation": "Company vacation policy: 20 days PTO per year. Carry over up to 5 days.",
        "remote": "Remote work policy: 3 days remote, 2 days in office per week.",
        "benefits": "Health insurance, 401k match up to 6%, gym membership, learning budget $2000/yr.",
        "onboarding": "New hire onboarding: 2 weeks. Buddy system. Complete security training by day 5.",
    }
    results = []
    for key, content in kb.items():
        if key in query.lower() or any(w in query.lower() for w in key.split()):
            results.append(content)
    return "\n".join(results) if results else f"No results found for: {query}"


@tool
def get_employee(name: str) -> str:
    """Look up an employee's info by name."""
    employees = {
        "alice": {"dept": "Engineering", "role": "Senior Dev", "manager": "Carol"},
        "bob": {"dept": "Marketing", "role": "Campaign Mgr", "manager": "Carol"},
        "carol": {"dept": "Engineering", "role": "VP Engineering", "manager": "CEO"},
    }
    emp = employees.get(name.lower())
    if emp:
        return json.dumps(emp)
    return f"Employee '{name}' not found."


tools = [calculator, search_knowledge, get_employee]
tool_map = {t.name: t for t in tools}

# ── State Definition ──────────────────────────────────────────
# State is a TypedDict. The `add_messages` annotation means new
# messages get APPENDED to the list (not replaced).


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Model ─────────────────────────────────────────────────────

model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024)
model_with_tools = model.bind_tools(tools)

# ── Node Functions ────────────────────────────────────────────
# Each node takes the state, does something, and returns updated state.


def agent_node(state: AgentState) -> AgentState:
    """Call the LLM. It will either respond with text or request tool calls."""
    print("\n── AGENT NODE ──")
    response = model_with_tools.invoke(state["messages"])

    # Log what the agent decided
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  Tool call: {tc['name']}({json.dumps(tc['args'])})")
    else:
        content = response.content
        if isinstance(content, str):
            print(f"  Final answer: {content[:100]}...")
        elif isinstance(content, list):
            text = next((b["text"] for b in content if b.get("type") == "text"), "")
            print(f"  Final answer: {text[:100]}...")

    return {"messages": [response]}


def tool_node(state: AgentState) -> AgentState:
    """Execute all tool calls from the last AI message."""
    print("\n── TOOL NODE ──")
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_fn = tool_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        print(f"  {tool_call['name']} → {str(result)[:100]}")
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}


# ── Conditional Edge ──────────────────────────────────────────
# After the agent node, decide: did it call tools, or is it done?


def should_continue(state: AgentState) -> str:
    """Route based on whether the agent requested tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ── Build the Graph ───────────────────────────────────────────

print("Building agent graph...")
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "agent")  # Start → agent
graph.add_conditional_edges(     # agent → tools OR end
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
graph.add_edge("tools", "agent")  # tools → back to agent

# Compile the graph into a runnable
agent = graph.compile()

# Visualize the graph structure
print("\nGraph structure:")
print("  START → agent")
print("  agent → tools (if tool_calls)")
print("  agent → END (if no tool_calls)")
print("  tools → agent (loop back)")

# ── Run It ────────────────────────────────────────────────────


def run_query(query: str):
    print(f"\n{'='*60}")
    print(f"USER: {query}")
    print(f"{'='*60}")

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # Extract final answer
    final = result["messages"][-1]
    content = final.content
    if isinstance(content, list):
        content = next((b["text"] for b in content if b.get("type") == "text"), "")
    print(f"\n✓ FINAL ANSWER: {content}")
    print(f"  Total messages in state: {len(result['messages'])}")
    return result


if __name__ == "__main__":
    # Test 1: Single tool call
    run_query("What's 256 * 789?")

    # Test 2: Multi-tool — agent needs to chain lookups
    run_query("Who is Alice's manager, and what's the vacation policy?")

    # Test 3: No tools needed
    run_query("Explain what a state graph is in one sentence.")

    # Test 4: Multi-step reasoning
    run_query(
        "Look up Bob's department, then search for info about the remote work policy. "
        "Finally, calculate how many remote days Bob gets in a year (52 weeks)."
    )
