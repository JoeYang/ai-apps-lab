"""
Session 10, Task 3: Checkpointing and Human-in-the-Loop.

Demonstrates:
1. Checkpointing — save and resume agent state across runs
2. Human-in-the-loop — pause the agent for approval before dangerous actions
3. Combining both — pause, persist, resume later
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ── Tools ─────────────────────────────────────────────────────

# Track actions for demonstration
action_log = []


@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    docs = {
        "refund": "Refund policy: Full refund within 30 days. 50% refund within 60 days. No refund after 60 days.",
        "pricing": "Enterprise plan: $500/month. Pro plan: $50/month. Free tier available.",
        "escalation": "Escalation: Contact support-lead@company.com for urgent issues.",
    }
    for key, content in docs.items():
        if key in query.lower():
            return content
    return "No documents found."


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This is a SENSITIVE action that requires approval."""
    action_log.append(f"EMAIL to={to} subject={subject}")
    return f"[SIMULATED] Email sent to {to}: {subject}"


@tool
def process_refund(customer_id: str, amount: float) -> str:
    """Process a refund for a customer. This is a SENSITIVE action that requires approval."""
    action_log.append(f"REFUND customer={customer_id} amount=${amount}")
    return f"[SIMULATED] Refund of ${amount} processed for customer {customer_id}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


tools = [search_docs, send_email, process_refund, calculator]
tool_map = {t.name: t for t in tools}

# Tools that require human approval before execution
SENSITIVE_TOOLS = {"send_email", "process_refund"}

# ── State ─────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Model ─────────────────────────────────────────────────────

model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024)
model_with_tools = model.bind_tools(tools)

# ── Nodes ─────────────────────────────────────────────────────


def agent_node(state: AgentState) -> AgentState:
    """Call the LLM."""
    print("\n── AGENT NODE ──")
    response = model_with_tools.invoke(state["messages"])
    if response.tool_calls:
        for tc in response.tool_calls:
            sensitive = " ⚠ SENSITIVE" if tc["name"] in SENSITIVE_TOOLS else ""
            print(f"  Tool call: {tc['name']}({json.dumps(tc['args'])}){sensitive}")
    else:
        content = response.content
        if isinstance(content, list):
            content = next((b["text"] for b in content if b.get("type") == "text"), "")
        print(f"  Final answer: {str(content)[:100]}...")
    return {"messages": [response]}


def tool_node(state: AgentState) -> AgentState:
    """Execute SAFE tool calls only."""
    print("\n── TOOL NODE (safe tools) ──")
    last_message = state["messages"][-1]
    results = []
    for tc in last_message.tool_calls:
        if tc["name"] not in SENSITIVE_TOOLS:
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"  ✓ {tc['name']} → {str(result)[:80]}")
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


def human_approval_node(state: AgentState) -> AgentState:
    """Execute SENSITIVE tool calls — in real system, this would pause for approval."""
    print("\n── HUMAN APPROVAL NODE ──")
    last_message = state["messages"][-1]
    results = []
    for tc in last_message.tool_calls:
        if tc["name"] in SENSITIVE_TOOLS:
            print(f"  🔒 Sensitive action: {tc['name']}({json.dumps(tc['args'])})")
            # In a real system with interrupt_before, the graph would PAUSE here.
            # The human would approve/deny via an external UI.
            # For this demo, we simulate automatic approval.
            print(f"  ✓ APPROVED (simulated)")
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"  Result: {str(result)[:80]}")
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


# ── Routing ───────────────────────────────────────────────────


def route_after_agent(state: AgentState) -> str:
    """Route based on what the agent wants to do."""
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "end"

    # Check if ANY tool call is sensitive
    has_sensitive = any(tc["name"] in SENSITIVE_TOOLS for tc in last.tool_calls)
    has_safe = any(tc["name"] not in SENSITIVE_TOOLS for tc in last.tool_calls)

    if has_sensitive and has_safe:
        return "both"
    elif has_sensitive:
        return "sensitive"
    else:
        return "safe"


# ── Build the Graph ───────────────────────────────────────────

print("Building graph with human-in-the-loop...\n")

graph = StateGraph(AgentState)

# Nodes
graph.add_node("agent", agent_node)
graph.add_node("safe_tools", tool_node)
graph.add_node("sensitive_tools", human_approval_node)

# Edges
graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "end": END,
        "safe": "safe_tools",
        "sensitive": "sensitive_tools",
        "both": "safe_tools",  # safe first, then sensitive
    },
)

graph.add_edge("safe_tools", "agent")
graph.add_edge("sensitive_tools", "agent")

# ── Part 1: Compile WITHOUT checkpointing ────────────────────

agent_no_memory = graph.compile()

print("Graph structure:")
print("  START → agent")
print("  agent → safe_tools (if only safe tools)")
print("  agent → sensitive_tools (if sensitive tools) ← HUMAN APPROVAL HERE")
print("  agent → END (if no tools)")
print("  safe_tools → agent")
print("  sensitive_tools → agent")

# ── Part 2: Compile WITH checkpointing ───────────────────────

checkpointer = MemorySaver()
agent_with_memory = graph.compile(checkpointer=checkpointer)

# ── Test Scenarios ────────────────────────────────────────────


def run_test(agent, query, label, thread_id=None):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"USER: {query}")
    print(f"{'='*60}")

    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]}, config=config
    )

    final = result["messages"][-1]
    content = final.content
    if isinstance(content, list):
        content = next((b["text"] for b in content if b.get("type") == "text"), str(content))
    print(f"\n✓ FINAL: {str(content)[:200]}")
    print(f"  Messages in state: {len(result['messages'])}")
    return result


if __name__ == "__main__":
    # Test 1: Safe query — no approval needed
    run_test(
        agent_no_memory,
        "What's our refund policy?",
        "Safe query — search docs only",
    )

    # Test 2: Sensitive action — requires approval
    run_test(
        agent_no_memory,
        "Customer C-123 wants a refund of $75. Process it and email them at customer@test.com to confirm.",
        "Sensitive action — refund + email",
    )

    # Test 3: Checkpointing — conversational memory across turns
    print("\n" + "█" * 60)
    print("CHECKPOINTING DEMO: Conversation memory")
    print("█" * 60)

    thread = "demo-thread-1"

    run_test(
        agent_with_memory,
        "What's our pricing?",
        "Turn 1 — ask about pricing",
        thread_id=thread,
    )

    run_test(
        agent_with_memory,
        "Which plan would you recommend for a 10-person startup?",
        "Turn 2 — follow-up (agent remembers pricing)",
        thread_id=thread,
    )

    run_test(
        agent_with_memory,
        "What were we just talking about?",
        "Turn 3 — test memory",
        thread_id=thread,
    )

    # Show that a DIFFERENT thread has no memory
    run_test(
        agent_with_memory,
        "What were we just talking about?",
        "Different thread — no shared memory",
        thread_id="different-thread",
    )

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("SUMMARY")
    print("█" * 60)
    print(f"\nActions logged: {action_log}")
    print("""
Key concepts demonstrated:

1. HUMAN-IN-THE-LOOP
   - Sensitive tools (send_email, process_refund) route to approval node
   - Safe tools (search_docs, calculator) execute immediately
   - In production: use `interrupt_before=["sensitive_tools"]` to truly pause

2. CHECKPOINTING
   - MemorySaver stores state after every node execution
   - Same thread_id = conversation continues with full history
   - Different thread_id = fresh conversation
   - In production: use SqliteSaver or PostgresSaver for persistence

3. WHY THIS IS HARD WITHOUT LANGGRAPH
   - Checkpointing: you'd need to serialize messages, tool state, turn count...
   - Human-in-the-loop: you'd need to save state, exit the loop, resume later
   - Routing: conditional logic for safe vs sensitive gets messy in a while loop
""")
