"""
Session 2, Task 4: Self-Consistency & Tree-of-Thought
======================================================
Two advanced prompting techniques for when you need the model
to reason carefully about complex problems.

SELF-CONSISTENCY:
  - Ask the model to solve the same problem multiple times
  - Each attempt may take a different reasoning path
  - Take the majority answer — if 4 out of 5 agree, that's likely correct
  - Like asking 5 engineers independently and going with the consensus

TREE-OF-THOUGHT (ToT):
  - Instead of one linear chain of reasoning, explore multiple branches
  - At each step, generate several possible next steps
  - Evaluate each branch before continuing
  - Like a chess player thinking several moves ahead

Run: python 04_advanced_reasoning.py
"""

import json
from collections import Counter

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5-20250929"


# ============================================================
# Problem: A complex trading incident that requires analysis
# ============================================================
INCIDENT = """
INCIDENT REPORT — 2026-02-16 09:30 UTC

Timeline:
- 09:28:00 Market data feed from NASDAQ shows latency spike to 450ms (normal: <10ms)
- 09:28:15 Matching engine CPU jumps from 52% to 89%
- 09:28:30 Risk engine starts rejecting orders: "stale market data" (142 rejects in 60s)
- 09:29:00 Client HEDGE_FUND_A margin utilisation hits 94%
- 09:29:15 FIX gateway detects sequence gap from HEDGE_FUND_A
- 09:29:30 HEDGE_FUND_A order rejection rate hits 23%
- 09:30:00 Market data latency returns to normal (8ms)
- 09:30:30 Matching engine CPU drops to 67%
- 09:31:00 Risk engine still rejecting — margin not recovered

Question: What is the root cause, and what is the correct remediation order?
"""


# ============================================================
# Self-Consistency: Ask N times, take majority answer
# ============================================================
def self_consistency(n_samples: int = 5):
    """
    Run the same analysis N times with temperature > 0.
    Each run may reason differently. The consensus answer
    is more reliable than any single answer.
    """
    print("=" * 60)
    print(f"SELF-CONSISTENCY ({n_samples} independent analyses)")
    print("=" * 60)

    root_causes = []
    full_responses = []

    for i in range(n_samples):
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            temperature=0.7,  # Some randomness so each attempt may differ
            messages=[{
                "role": "user",
                "content": f"""{INCIDENT}

Analyse this incident. Answer these two questions concisely:
1. ROOT CAUSE (one sentence): What triggered the cascade?
2. FIRST REMEDIATION STEP: What should be done first?

Format your answer as:
ROOT_CAUSE: <your answer>
FIRST_STEP: <your answer>""",
            }],
        )

        text = response.content[0].text
        full_responses.append(text)

        # Extract root cause line
        for line in text.split("\n"):
            if line.strip().startswith("ROOT_CAUSE:"):
                root_causes.append(line.strip())
                break

        print(f"\n--- Sample {i+1} ---")
        print(text.strip())

    # Find consensus
    print("\n" + "=" * 60)
    print("CONSENSUS ANALYSIS:")
    print("=" * 60)
    print(f"\nAll root causes identified:")
    for i, rc in enumerate(root_causes):
        print(f"  {i+1}. {rc}")

    print(f"\nWith {n_samples} independent analyses, look for agreement.")
    print("If most analyses point to the same root cause, that's your answer.")
    print("If they diverge, the problem may be ambiguous and needs more data.")


# ============================================================
# Tree-of-Thought: Explore multiple reasoning branches
# ============================================================
def tree_of_thought():
    """
    Instead of one linear reasoning chain, we:
    1. Generate multiple possible hypotheses
    2. Evaluate each hypothesis against the evidence
    3. Select the best-supported hypothesis
    4. Determine remediation based on the winning hypothesis

    This is like a structured investigation — consider all possibilities
    before committing to an answer.
    """
    print("\n" + "=" * 60)
    print("TREE-OF-THOUGHT (structured investigation)")
    print("=" * 60)

    # Step 1: Generate hypotheses
    print("\n--- Step 1: Generate hypotheses ---")
    step1 = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""{INCIDENT}

You are investigating this incident. Generate exactly 3 different hypotheses
for the root cause. For each, list the evidence that supports AND contradicts it.

Format:
HYPOTHESIS 1: <description>
  Supporting evidence: <bullets>
  Contradicting evidence: <bullets>

HYPOTHESIS 2: ...
HYPOTHESIS 3: ...""",
        }],
    )
    hypotheses = step1.content[0].text
    print(hypotheses)

    # Step 2: Evaluate each hypothesis
    print("\n--- Step 2: Evaluate and score hypotheses ---")
    step2 = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Given this incident and these hypotheses:

{INCIDENT}

{hypotheses}

Score each hypothesis from 1-10 on:
- Evidence fit (how well does it explain ALL events in the timeline?)
- Simplicity (does it require the fewest assumptions?)
- Actionability (does it lead to a clear remediation?)

Format as a table, then declare a WINNER with justification.""",
        }],
    )
    evaluation = step2.content[0].text
    print(evaluation)

    # Step 3: Remediation plan based on winning hypothesis
    print("\n--- Step 3: Remediation plan ---")
    step3 = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Based on this analysis:

{evaluation}

Provide a numbered remediation plan in priority order.
For each step, specify:
- What to do
- Why this must come before the next step
- Expected outcome
- Who should execute it (e.g., network team, risk team, support)""",
        }],
    )
    print(step3.content[0].text)


# ============================================================
# Comparison: Simple CoT vs Tree-of-Thought
# ============================================================
def simple_cot():
    """Basic chain-of-thought for comparison."""
    print("\n" + "=" * 60)
    print("BASIC CHAIN-OF-THOUGHT (for comparison)")
    print("=" * 60)

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""{INCIDENT}

Think step by step to determine the root cause and remediation plan.""",
        }],
    )
    print(response.content[0].text)


# --- Run all techniques ---
if __name__ == "__main__":
    print("SESSION 2: Advanced Reasoning Techniques")
    print("All techniques applied to the SAME trading incident\n")

    simple_cot()
    tree_of_thought()
    self_consistency(n_samples=3)  # 3 samples to save API cost

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("  SELF-CONSISTENCY:")
    print("    - Multiple independent attempts → majority vote")
    print("    - Costs N times more, but catches reasoning errors")
    print("    - Best for: high-stakes decisions, ambiguous problems")
    print()
    print("  TREE-OF-THOUGHT:")
    print("    - Generate hypotheses → evaluate → select → plan")
    print("    - Explores alternatives before committing to an answer")
    print("    - Best for: investigations, diagnosis, complex analysis")
    print()
    print("  BASIC CoT:")
    print("    - Single linear reasoning chain")
    print("    - Cheapest, fastest, good enough for many tasks")
    print("    - Risk: can go down the wrong path and not recover")
    print("=" * 60)
