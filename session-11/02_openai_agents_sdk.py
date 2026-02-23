"""
Session 11, Task 3: OpenAI Agents SDK — multi-agent via handoffs.

The OpenAI Agents SDK uses a different model from LangGraph:
- Agents are defined with instructions, tools, and HANDOFFS
- Handoffs transfer control from one agent to another
- No explicit graph — the LLM decides when to hand off
- Built-in tracing and guardrails

We'll build the same researcher → writer → reviewer pipeline.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, function_tool, handoff, trace

# ── Tools ─────────────────────────────────────────────────────


@function_tool
def search_web(query: str) -> str:
    """Search the web for information on a topic."""
    results = {
        "react agent": (
            "ReAct combines chain-of-thought reasoning with tool use. "
            "Introduced by Yao et al. (2023). Agents think, act, observe in a loop. "
            "Outperforms CoT alone by ~30% on knowledge tasks."
        ),
        "multi-agent": (
            "Multi-agent patterns: supervisor (one delegates), hierarchical, "
            "collaborative (peer-to-peer), debate (adversarial). "
            "AutoGen showed 25% improvement in code gen via debate."
        ),
        "tool use": (
            "Modern LLMs support native tool/function calling via structured APIs. "
            "Claude, GPT-4, Gemini all support tool_use content blocks. "
            "Key: schema definition, execution, result feeding."
        ),
    }
    query_lower = query.lower()
    for key, content in results.items():
        if key in query_lower or any(w in query_lower for w in key.split()):
            return content
    return f"General information available about '{query}'."


@function_tool
def search_papers(query: str) -> str:
    """Search academic papers on a topic."""
    papers = {
        "react": "Yao et al. (2023) 'ReAct: Synergizing Reasoning and Acting' — ReAct outperforms CoT on knowledge-intensive tasks.",
        "agent": "Wang et al. (2024) 'Survey on LLM-based Agents' — comprehensive coverage of planning, memory, tool use.",
        "multi-agent": "Wu et al. (2023) 'AutoGen' — multi-agent debate improves code gen accuracy by 25%.",
    }
    results = []
    for key, paper in papers.items():
        if key in query.lower():
            results.append(paper)
    return "\n".join(results) if results else "No papers found."


@function_tool
def check_facts(claim: str) -> str:
    """Verify a factual claim for accuracy."""
    verified = {
        "react": "VERIFIED: ReAct was introduced by Yao et al. in 2023.",
        "30%": "PARTIALLY VERIFIED: Improvement varies by task, but ReAct consistently outperforms CoT.",
        "25%": "VERIFIED: AutoGen showed ~25% improvement on code generation.",
    }
    for key, result in verified.items():
        if key in claim.lower():
            return result
    return "UNVERIFIED: Recommend citing primary sources."


# ── Define Agents ─────────────────────────────────────────────

# Note: We define agents bottom-up so handoff targets exist first.

reviewer = Agent(
    name="Reviewer",
    instructions="""You are a critical reviewer. Evaluate the article for:
1. Factual accuracy — use check_facts to verify key claims
2. Completeness and clarity
3. Structure

If the article is good, respond with your approval and final version.
If it needs work, provide specific feedback.""",
    tools=[check_facts],
    # Reviewer is the last agent — no handoffs
)

writer = Agent(
    name="Writer",
    instructions="""You are a professional technical writer.
Write a clear, well-structured article based on the research provided in the conversation.
Use headers, bullet points, and include a conclusion.
When done writing, hand off to the Reviewer for feedback.""",
    handoffs=[reviewer],
    # Writer has no tools — just writes
)

researcher = Agent(
    name="Researcher",
    instructions="""You are a research specialist. Gather comprehensive information on the topic.
Use search_web and search_papers to find relevant information.
Synthesise findings into a clear research summary with citations.
When research is complete, hand off to the Writer.""",
    tools=[search_web, search_papers],
    handoffs=[writer],
)

# ── Triage agent (like our supervisor) ────────────────────────

triage_agent = Agent(
    name="Triage",
    instructions="""You are the coordinator. When you receive a request to write an article,
hand off to the Researcher to begin the process.
The pipeline is: Researcher → Writer → Reviewer.""",
    handoffs=[researcher],
)

# ── Run It ────────────────────────────────────────────────────


async def main():
    print("=" * 60)
    print("OpenAI Agents SDK: Multi-Agent Article Pipeline")
    print("=" * 60)
    print("\nPipeline: Triage → Researcher → Writer → Reviewer\n")

    result = await Runner.run(
        triage_agent,
        "Write a short article about modern AI agent architectures — cover ReAct, tool use, and multi-agent patterns.",
    )

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(f"\nFinal agent: {result.last_agent.name}")
    print(f"\n{result.final_output}")

    # Show the agent handoff chain
    print("\n" + "=" * 60)
    print("AGENT TRACE")
    print("=" * 60)
    for i, item in enumerate(result.raw_responses):
        print(f"  Step {i+1}: {item.response.model if hasattr(item, 'response') else 'unknown'}")

    print(f"\n  New items in conversation: {len(result.new_items)}")
    for item in result.new_items:
        item_type = type(item).__name__
        if hasattr(item, "agent"):
            print(f"    {item_type} (agent: {item.agent.name})")
        elif hasattr(item, "source_agent"):
            print(f"    {item_type}: {item.source_agent.name} → {item.target_agent.name}")
        else:
            print(f"    {item_type}")


if __name__ == "__main__":
    asyncio.run(main())
