"""
Session 10, Task 4: LangGraph vs Simple Loop comparison.

Implements the SAME agent task both ways, then compares:
- Lines of code
- Flexibility
- Feature support

Task: A support agent that looks up info, and requires approval for refunds.
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

import anthropic
from typing import Annotated
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ── Shared tool logic ─────────────────────────────────────────

KNOWLEDGE = {
    "refund": "Refund policy: Full refund within 30 days.",
    "pricing": "Pro: $50/mo. Enterprise: $500/mo.",
}

EMPLOYEES = {
    "alice": {"dept": "Engineering", "role": "Senior Dev"},
}

SENSITIVE = {"process_refund"}


def _search(query):
    for k, v in KNOWLEDGE.items():
        if k in query.lower():
            return v
    return "No results."


def _refund(customer_id, amount):
    return f"Refund ${amount} processed for {customer_id}"


# ══════════════════════════════════════════════════════════════
# APPROACH 1: SIMPLE LOOP (Session 9 style)
# ══════════════════════════════════════════════════════════════


def run_simple_loop(query: str) -> str:
    """ReAct agent as a simple while loop."""
    client = anthropic.Anthropic()

    tools = [
        {
            "name": "search_docs",
            "description": "Search documentation.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "process_refund",
            "description": "Process a refund. Requires approval.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["customer_id", "amount"],
            },
        },
    ]

    tool_fns = {
        "search_docs": lambda inp: _search(inp["query"]),
        "process_refund": lambda inp: _refund(inp["customer_id"], inp["amount"]),
    }

    messages = [{"role": "user", "content": query}]

    for turn in range(10):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system="You are a support agent. Be concise.",
            tools=tools,
            messages=messages,
        )

        content = response.content
        tool_results = []

        for block in content:
            if block.type == "tool_use":
                # Manual approval check
                if block.name in SENSITIVE:
                    print(f"  [SIMPLE] 🔒 Approval needed: {block.name}")
                    # Simulate approval
                    approved = True
                    if not approved:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "DENIED by human.",
                        })
                        continue

                result = tool_fns[block.name](block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "assistant", "content": content})

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "end_turn":
            return next((b.text for b in content if b.type == "text"), "")

    return "Max turns."


# ══════════════════════════════════════════════════════════════
# APPROACH 2: LANGGRAPH
# ══════════════════════════════════════════════════════════════


@lc_tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return _search(query)


@lc_tool
def process_refund(customer_id: str, amount: float) -> str:
    """Process a refund. Requires approval."""
    return _refund(customer_id, amount)


lg_tools = [search_docs, process_refund]
lg_tool_map = {t.name: t for t in lg_tools}


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=512)
model_with_tools = model.bind_tools(lg_tools)


def agent_node(state):
    return {"messages": [model_with_tools.invoke(state["messages"])]}


def tool_node(state):
    """Execute all tool calls — flag sensitive ones for approval."""
    results = []
    for tc in state["messages"][-1].tool_calls:
        if tc["name"] in SENSITIVE:
            print(f"  [GRAPH] 🔒 Approval needed: {tc['name']}")
            # In production: interrupt_before would pause here
        r = lg_tool_map[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(r), tool_call_id=tc["id"]))
    return {"messages": results}


def route(state):
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "end"
    return "tools"


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

checkpointer = MemorySaver()
lg_agent = graph.compile(checkpointer=checkpointer)


def run_langgraph(query: str, thread_id: str = "t1") -> str:
    result = lg_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    final = result["messages"][-1]
    content = final.content
    if isinstance(content, list):
        content = next((b["text"] for b in content if b.get("type") == "text"), str(content))
    return content


# ══════════════════════════════════════════════════════════════
# RUN BOTH AND COMPARE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    query = "What's the refund policy? Then process a $50 refund for customer C-99."

    print("=" * 60)
    print("APPROACH 1: SIMPLE LOOP")
    print("=" * 60)
    t1 = time.time()
    result1 = run_simple_loop(query)
    t1 = time.time() - t1
    print(f"\nResult: {result1[:200]}")
    print(f"Time: {t1:.1f}s")

    print(f"\n{'='*60}")
    print("APPROACH 2: LANGGRAPH")
    print("=" * 60)
    t2 = time.time()
    result2 = run_langgraph(query)
    t2 = time.time() - t2
    print(f"\nResult: {result2[:200]}")
    print(f"Time: {t2:.1f}s")

    # ── Demonstrate what LangGraph gives you for free ────────
    print(f"\n{'='*60}")
    print("WHAT LANGGRAPH ADDS FOR FREE")
    print("=" * 60)

    # Conversational memory
    print("\n1. CONVERSATIONAL MEMORY:")
    r = run_langgraph("What did I just ask about?", thread_id="t1")
    print(f"   Follow-up answer: {r[:150]}")

    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"""
┌──────────────────────────┬──────────────────┬──────────────────┐
│ Feature                  │ Simple Loop      │ LangGraph        │
├──────────────────────────┼──────────────────┼──────────────────┤
│ Basic agent loop         │ ✓ Easy           │ ✓ More setup     │
│ Tool calling             │ ✓ Manual parsing │ ✓ Automatic      │
│ Human-in-the-loop        │ ~ DIY in loop    │ ✓ interrupt_before│
│ Checkpointing/memory     │ ✗ Build yourself │ ✓ One line       │
│ Multi-agent handoffs     │ ✗ Very hard      │ ✓ Built-in       │
│ Conditional routing      │ ~ if/else        │ ✓ Graph edges    │
│ Visualization            │ ✗ None           │ ✓ Graph diagram  │
│ Streaming                │ ~ Manual         │ ✓ Built-in       │
│ Debugging/tracing        │ ~ Manual logging │ ✓ LangSmith      │
│ Code clarity             │ ✓ Obvious flow   │ ~ Abstracted     │
│ Learning curve           │ ✓ Low            │ ~ Medium         │
│ Vendor lock-in           │ ✓ None           │ ~ LangChain      │
│ Control                  │ ✓ Full           │ ~ Framework      │
├──────────────────────────┴──────────────────┴──────────────────┤
│                                                                │
│ WHEN TO USE SIMPLE LOOP:                                       │
│   • Prototyping / learning                                     │
│   • Single agent, linear tool use                              │
│   • You want full control and minimal dependencies             │
│   • Simple use case that won't grow much                       │
│                                                                │
│ WHEN TO USE LANGGRAPH:                                         │
│   • Need persistence (pause/resume across sessions)            │
│   • Need human-in-the-loop for sensitive actions               │
│   • Multi-agent coordination                                   │
│   • Complex branching / conditional workflows                  │
│   • Production system with observability requirements          │
│   • Team project (graph is self-documenting)                   │
│                                                                │
│ THE RULE OF THUMB:                                             │
│   Start with the simple loop.                                  │
│   Switch to LangGraph when you need a feature it provides.     │
│   Don't adopt a framework "just in case."                      │
└────────────────────────────────────────────────────────────────┘
""")
