"""
Session 11, Task 2: Multi-agent system with LangGraph.

Supervisor pattern: a supervisor delegates to specialised agents.
  - Researcher: searches for information
  - Writer: produces content from research
  - Reviewer: critiques and suggests improvements

The supervisor decides who works next and when to stop.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ── Tools (each agent gets different tools) ───────────────────


@tool
def search_web(query: str) -> str:
    """Search the web for information on a topic."""
    # Simulated search results
    results = {
        "react agent": (
            "ReAct (Reasoning + Acting) combines chain-of-thought reasoning with "
            "tool use. The agent thinks step-by-step, takes actions, and observes "
            "results. Introduced by Yao et al. (2023). Used in LangChain, Claude, "
            "and most modern agent frameworks. Key advantage: grounds reasoning in "
            "real-world observations, reducing hallucination."
        ),
        "langgraph": (
            "LangGraph is a framework for building stateful, multi-actor applications "
            "with LLMs. Built on LangChain. Key concepts: state graphs, nodes, edges, "
            "checkpointing. Supports human-in-the-loop, persistence, and streaming. "
            "Used for complex agent workflows."
        ),
        "multi-agent": (
            "Multi-agent systems coordinate multiple LLM agents. Common patterns: "
            "supervisor (one agent delegates), hierarchical (multi-level), "
            "collaborative (peer-to-peer), and debate (adversarial). Benefits: "
            "specialisation, separation of concerns, parallel work. Costs: more "
            "API calls, latency, complexity. Best when tasks naturally decompose."
        ),
        "ai agent": (
            "AI agents are LLM-powered systems that can take actions autonomously. "
            "Key components: reasoning (LLM), tools (APIs/functions), memory "
            "(conversation history, vector stores), and planning (task decomposition). "
            "Challenges: reliability, cost, hallucination, looping."
        ),
    }
    query_lower = query.lower()
    for key, content in results.items():
        if key in query_lower or any(w in query_lower for w in key.split()):
            return content
    return f"Search results for '{query}': General information about this topic is available in AI research papers and documentation."


@tool
def search_papers(query: str) -> str:
    """Search academic papers for research on a topic."""
    papers = {
        "react": "Yao et al. (2023) 'ReAct: Synergizing Reasoning and Acting in Language Models' — showed ReAct outperforms chain-of-thought alone on knowledge-intensive tasks by 30%.",
        "agent": "Wang et al. (2024) 'A Survey on LLM-based Agents' — comprehensive survey of agent architectures, covering planning, memory, and tool use patterns.",
        "multi-agent": "Wu et al. (2023) 'AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation' — demonstrated that multi-agent debate improves code generation accuracy by 25%.",
    }
    results = []
    for key, paper in papers.items():
        if key in query.lower():
            results.append(paper)
    return "\n".join(results) if results else "No relevant papers found."


@tool
def check_facts(claim: str) -> str:
    """Verify a factual claim. Returns whether it's accurate."""
    # Simulated fact checker
    verified = {
        "react": "VERIFIED: ReAct was introduced by Yao et al. in 2023.",
        "langgraph": "VERIFIED: LangGraph is built on LangChain and uses state graphs.",
        "30%": "PARTIALLY VERIFIED: The exact improvement varies by task, but ReAct does consistently outperform CoT alone.",
        "25%": "VERIFIED: AutoGen multi-agent debate showed ~25% improvement on code generation benchmarks.",
    }
    for key, result in verified.items():
        if key in claim.lower():
            return result
    return "UNVERIFIED: Could not independently verify this claim. Recommend citing sources."


# ── Models ────────────────────────────────────────────────────

# Using the same model for all agents, but in production you might use
# a cheaper model for research and a more capable one for writing.
model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024)

researcher_model = model.bind_tools([search_web, search_papers])
reviewer_model = model.bind_tools([check_facts])
# Writer has no tools — just writes from the research provided

# ── State ─────────────────────────────────────────────────────


class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research: str           # accumulated research findings
    draft: str              # current article draft
    review: str             # reviewer feedback
    next_agent: str         # who should work next
    iteration: int          # track revision cycles


# ── Agent Nodes ───────────────────────────────────────────────

RESEARCHER_SYSTEM = """You are a research specialist. Your job is to gather comprehensive information on the given topic.
Use your search tools to find relevant information and academic papers.
Synthesise your findings into a clear research summary.
Always cite your sources."""

WRITER_SYSTEM = """You are a professional technical writer. Your job is to write a clear, well-structured article based on the research provided.
Write in a professional but accessible tone.
Structure with headers, bullet points where appropriate, and a conclusion.
If reviewer feedback is provided, revise accordingly.
Output ONLY the article text — no meta-commentary."""

REVIEWER_SYSTEM = """You are a critical reviewer. Your job is to evaluate the article for:
1. Factual accuracy — use check_facts tool to verify key claims
2. Completeness — does it cover the topic adequately?
3. Clarity — is it well-written and understandable?
4. Structure — is it well-organised?

Provide specific, actionable feedback. If the article is good enough, say "APPROVED" at the start of your response."""


def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """Decide which agent should work next."""
    print("\n── SUPERVISOR ──")

    iteration = state.get("iteration", 0)
    research = state.get("research", "")
    draft = state.get("draft", "")
    review = state.get("review", "")

    if not research:
        next_agent = "researcher"
        print(f"  Decision: No research yet → sending to RESEARCHER")
    elif not draft:
        next_agent = "writer"
        print(f"  Decision: Research done, no draft → sending to WRITER")
    elif not review:
        next_agent = "reviewer"
        print(f"  Decision: Draft done, not reviewed → sending to REVIEWER")
    elif "APPROVED" in review:
        next_agent = "done"
        print(f"  Decision: Review APPROVED → DONE")
    elif iteration >= 2:
        next_agent = "done"
        print(f"  Decision: Max iterations ({iteration}) → DONE (accepting current draft)")
    else:
        next_agent = "writer"
        print(f"  Decision: Revision needed (iteration {iteration + 1}) → sending to WRITER")

    return {"next_agent": next_agent}


def researcher_node(state: MultiAgentState) -> MultiAgentState:
    """Research agent gathers information."""
    print("\n── RESEARCHER ──")

    # Get the original user request
    user_msg = state["messages"][0].content

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=f"Research this topic thoroughly: {user_msg}"),
    ]

    # Run researcher with tools (may take multiple turns)
    research_notes = []
    for turn in range(5):
        response = researcher_model.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            from langchain_core.messages import ToolMessage
            for tc in response.tool_calls:
                tool_fn = {"search_web": search_web, "search_papers": search_papers}[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                print(f"  Tool: {tc['name']}({tc['args']}) → {str(result)[:80]}...")
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            # Agent is done researching
            content = response.content
            if isinstance(content, list):
                content = next((b["text"] for b in content if b.get("type") == "text"), "")
            research_notes.append(content)
            break

    research = "\n".join(research_notes)
    print(f"  Research summary: {research[:100]}...")
    return {"research": research}


def writer_node(state: MultiAgentState) -> MultiAgentState:
    """Writer agent produces/revises content."""
    print("\n── WRITER ──")

    user_msg = state["messages"][0].content
    research = state.get("research", "")
    review = state.get("review", "")
    old_draft = state.get("draft", "")

    prompt = f"Topic: {user_msg}\n\nResearch notes:\n{research}"
    if review and old_draft:
        prompt += f"\n\nPrevious draft:\n{old_draft}\n\nReviewer feedback:\n{review}\n\nPlease revise the article based on the feedback."

    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = model.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = next((b["text"] for b in content if b.get("type") == "text"), "")

    print(f"  Draft written: {content[:100]}...")
    return {"draft": content, "iteration": state.get("iteration", 0) + 1}


def reviewer_node(state: MultiAgentState) -> MultiAgentState:
    """Reviewer agent critiques the draft."""
    print("\n── REVIEWER ──")

    draft = state.get("draft", "")

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"Review this article:\n\n{draft}"),
    ]

    # Run reviewer with fact-checking tool
    for turn in range(3):
        response = reviewer_model.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            from langchain_core.messages import ToolMessage
            for tc in response.tool_calls:
                result = check_facts.invoke(tc["args"])
                print(f"  Fact check: {tc['args']['claim'][:60]}... → {str(result)[:60]}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            content = response.content
            if isinstance(content, list):
                content = next((b["text"] for b in content if b.get("type") == "text"), "")
            approved = "APPROVED" in content
            print(f"  Review: {'APPROVED ✓' if approved else 'REVISION NEEDED'}")
            print(f"  Feedback: {content[:100]}...")
            return {"review": content}

    return {"review": "APPROVED (max review turns reached)"}


# ── Routing ───────────────────────────────────────────────────


def route_from_supervisor(state: MultiAgentState) -> str:
    return state.get("next_agent", "researcher")


# ── Build the Graph ───────────────────────────────────────────

print("Building multi-agent graph...\n")

graph = StateGraph(MultiAgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "writer": "writer",
        "reviewer": "reviewer",
        "done": END,
    },
)

# All workers report back to supervisor
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
graph.add_edge("reviewer", "supervisor")

agent = graph.compile()

print("Graph structure:")
print("  START → supervisor")
print("  supervisor → researcher | writer | reviewer | END")
print("  researcher → supervisor")
print("  writer → supervisor")
print("  reviewer → supervisor")
print("  (supervisor decides who goes next each cycle)\n")

# ── Run It ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TASK: Write an article about AI agent architectures")
    print("=" * 60)

    result = agent.invoke({
        "messages": [HumanMessage(content="Write a short article about modern AI agent architectures — cover ReAct, tool use, and multi-agent patterns.")],
        "research": "",
        "draft": "",
        "review": "",
        "next_agent": "",
        "iteration": 0,
    })

    print("\n" + "=" * 60)
    print("FINAL ARTICLE")
    print("=" * 60)
    print(result["draft"])

    print("\n" + "=" * 60)
    print("PROCESS SUMMARY")
    print("=" * 60)
    print(f"  Iterations: {result['iteration']}")
    print(f"  Review status: {'APPROVED' if 'APPROVED' in result.get('review', '') else 'Accepted at max iterations'}")
    print(f"  Research length: {len(result.get('research', ''))} chars")
    print(f"  Final draft length: {len(result.get('draft', ''))} chars")
