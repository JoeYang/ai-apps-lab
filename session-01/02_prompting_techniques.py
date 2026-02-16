"""
Session 1, Task 3: Prompting Techniques
========================================
Compare zero-shot, few-shot, role-based, and chain-of-thought prompting
on the SAME task to see the difference in output quality.

Task: Classify trading system log messages by severity and category.

Run: python 02_prompting_techniques.py
"""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5-20250929"

# Sample log messages to classify
LOG_MESSAGES = [
    "2026-02-15 09:30:01.234 Order 12345 rejected: insufficient margin for AAPL 1000@185.50",
    "2026-02-15 09:30:01.567 FIX session SENDER1-TARGET1 heartbeat timeout after 35s",
    "2026-02-15 09:30:02.001 Market data latency spike: NASDAQ feed 450ms (threshold: 100ms)",
    "2026-02-15 09:30:02.123 Successfully connected to exchange gateway on backup port 9443",
    "2026-02-15 09:30:03.456 Position limit breach: TSLA net position 50,200 exceeds limit 50,000",
]

LOGS_BLOCK = "\n".join(f"{i+1}. {msg}" for i, msg in enumerate(LOG_MESSAGES))


def zero_shot():
    """
    ZERO-SHOT: Just ask the model to do the task with no examples.
    The model relies entirely on its training knowledge.
    Works for simple, well-understood tasks. Can be inconsistent in format.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Classify these trading system log messages by severity (critical/warning/info) and category:\n\n{LOGS_BLOCK}",
        }],
    )
    return response.content[0].text


def few_shot():
    """
    FEW-SHOT: Provide examples of the desired input→output mapping.
    The model learns the pattern from your examples and applies it.
    Much more consistent output format. Better accuracy on edge cases.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Classify trading system log messages by severity and category.

Here are some examples:

Log: "Order 99999 filled: BUY MSFT 500@320.00 in 0.3ms"
Classification: severity=INFO, category=ORDER_EXECUTION, action=NONE

Log: "FIX session disconnected: SENDER2-TARGET2, attempting reconnect"
Classification: severity=CRITICAL, category=CONNECTIVITY, action=PAGE_ONCALL

Log: "CPU usage on matching engine server at 78%"
Classification: severity=WARNING, category=SYSTEM_RESOURCE, action=MONITOR

Now classify these messages:

{LOGS_BLOCK}""",
        }],
    )
    return response.content[0].text


def role_based():
    """
    ROLE-BASED: Assign the model a specific persona via the system prompt.
    The model adopts domain expertise, vocabulary, and priorities of that role.
    Produces more contextually appropriate responses.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="""You are a Level 3 trading systems support engineer with 10 years of experience
in equities electronic trading. You monitor production systems for a major broker-dealer.
You know that FIX connectivity issues and position limit breaches require immediate escalation,
while order rejections for margin are usually handled by the risk team during normal hours.
Classify log messages and recommend specific actions based on your operational experience.""",
        messages=[{
            "role": "user",
            "content": f"Classify these production alerts that just came in:\n\n{LOGS_BLOCK}",
        }],
    )
    return response.content[0].text


def chain_of_thought():
    """
    CHAIN-OF-THOUGHT (CoT): Ask the model to reason step by step before answering.
    Forces the model to "show its work", leading to more accurate conclusions.
    Especially useful for tasks requiring analysis or judgement.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Classify these trading system log messages by severity and category.

For each message, think step by step:
1. What system component is involved?
2. What happened — is it a failure, degradation, or normal operation?
3. What's the potential business impact if unaddressed?
4. Based on the above, assign severity (CRITICAL/WARNING/INFO) and category.

Messages:

{LOGS_BLOCK}""",
        }],
    )
    return response.content[0].text


# --- Run and compare all techniques ---
if __name__ == "__main__":
    techniques = [
        ("ZERO-SHOT", zero_shot),
        ("FEW-SHOT", few_shot),
        ("ROLE-BASED", role_based),
        ("CHAIN-OF-THOUGHT", chain_of_thought),
    ]

    for name, func in techniques:
        print("=" * 60)
        print(f"  {name}")
        print("=" * 60)
        result = func()
        print(result)
        print()

    print("=" * 60)
    print("COMPARE THE OUTPUTS ABOVE. Notice:")
    print("  - Zero-shot: works but format may vary")
    print("  - Few-shot: consistent format matching your examples")
    print("  - Role-based: domain-specific language and prioritisation")
    print("  - Chain-of-thought: explicit reasoning, better for complex judgements")
    print()
    print("WHEN TO USE EACH:")
    print("  Zero-shot   → simple tasks, prototyping")
    print("  Few-shot    → need consistent output format")
    print("  Role-based  → need domain expertise and appropriate tone")
    print("  CoT         → complex reasoning, analysis, multi-step logic")
    print("  Combined    → production systems often mix all of these!")
    print("=" * 60)
