"""
Session 2, Task 1: Structured Outputs
======================================
The problem: LLMs return free-form text. In production, you need
reliable, parseable data structures — not prose.

This script shows 3 approaches, from basic to production-grade:
  1. Prompt-based JSON (fragile)
  2. JSON with manual validation (better)
  3. Pydantic models with automatic validation (production-grade)

Run: python 01_structured_outputs.py
"""

import json

import anthropic
from pydantic import BaseModel, Field, ValidationError

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5-20250929"

# Sample trading alert for all examples
ALERT = """
2026-02-16 09:30:01.234 ALERT: Order rejection rate for client HEDGE_FUND_A
spiked to 23% (threshold: 5%) over the last 60 seconds. 142 out of 617 orders
rejected. Primary rejection reason: "insufficient buying power". Affected symbols:
TSLA, NVDA, AAPL. Client is running a momentum strategy with high order frequency.
Risk system flagged margin utilisation at 94%.
"""


# ============================================================
# Approach 1: Prompt-based JSON (fragile)
# ============================================================
def approach_1_prompt_only():
    """
    Just ask the model to return JSON. This works... sometimes.
    Problems: model may add markdown fences, extra text, or vary the schema.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Analyse this trading alert and return a JSON object with these fields:
- severity (critical/warning/info)
- category (string)
- affected_client (string)
- affected_symbols (list of strings)
- root_cause (string)
- recommended_action (string)

Alert: {ALERT}

Return ONLY the JSON, no other text.""",
        }],
    )

    text = response.content[0].text
    print("=== Approach 1: Prompt-only JSON ===")
    print(f"Raw response:\n{text}\n")

    # Try to parse it — this may fail if the model adds markdown fences etc.
    try:
        # Strip markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(cleaned)
        print(f"Parsed successfully: {list(data.keys())}")
    except json.JSONDecodeError as e:
        print(f"PARSE FAILED: {e}")

    print()
    return text


# ============================================================
# Approach 2: JSON with manual validation (better)
# ============================================================
def approach_2_manual_validation():
    """
    Use the system prompt to enforce JSON format, then validate manually.
    More reliable but verbose and error-prone to maintain.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="""You are a trading alert classifier. You ALWAYS respond with valid JSON only.
Never include markdown fences, explanations, or any text outside the JSON object.
Your response must be parseable by json.loads() directly.""",
        messages=[{
            "role": "user",
            "content": f"""Classify this alert. Required JSON schema:
{{
    "severity": "critical" | "warning" | "info",
    "category": string,
    "affected_client": string,
    "affected_symbols": [string],
    "root_cause": string,
    "recommended_action": string,
    "requires_immediate_escalation": boolean
}}

Alert: {ALERT}""",
        }],
    )

    text = response.content[0].text
    print("=== Approach 2: Manual Validation ===")

    try:
        data = json.loads(text)

        # Manual validation — tedious and error-prone
        required_fields = ["severity", "category", "affected_client",
                           "affected_symbols", "root_cause", "recommended_action",
                           "requires_immediate_escalation"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            print(f"VALIDATION FAILED: missing fields: {missing}")
        elif data["severity"] not in ("critical", "warning", "info"):
            print(f"VALIDATION FAILED: invalid severity: {data['severity']}")
        elif not isinstance(data["affected_symbols"], list):
            print(f"VALIDATION FAILED: affected_symbols must be a list")
        else:
            print(f"Valid! severity={data['severity']}, client={data['affected_client']}")
            print(f"  symbols={data['affected_symbols']}")
            print(f"  escalate={data['requires_immediate_escalation']}")

    except json.JSONDecodeError as e:
        print(f"PARSE FAILED: {e}")

    print()
    return text


# ============================================================
# Approach 3: Pydantic models (production-grade)
# ============================================================

# Define your schema as a Pydantic model — this is your contract
class AlertClassification(BaseModel):
    """Structured classification of a trading system alert."""
    severity: str = Field(description="One of: critical, warning, info")
    category: str = Field(description="Alert category, e.g., risk, connectivity, latency")
    affected_client: str = Field(description="Client identifier")
    affected_symbols: list[str] = Field(description="List of affected ticker symbols")
    root_cause: str = Field(description="Brief root cause analysis")
    recommended_action: str = Field(description="What the support team should do")
    requires_immediate_escalation: bool = Field(description="Whether to page on-call immediately")
    confidence: float = Field(description="Confidence score 0.0 to 1.0", ge=0.0, le=1.0)


def approach_3_pydantic():
    """
    Use Pydantic to:
    1. Generate the JSON schema automatically from the model definition
    2. Validate the response automatically (types, constraints, required fields)
    3. Get proper Python objects with autocomplete and type safety

    This is the production pattern. Your Pydantic model IS the contract
    between your LLM and the rest of your application.
    """

    # Generate JSON schema from Pydantic model — no manual schema writing!
    schema = AlertClassification.model_json_schema()

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="""You are a trading alert classifier. Respond with valid JSON only.
No markdown fences, no explanations — just the JSON object.""",
        messages=[{
            "role": "user",
            "content": f"""Classify this alert according to this exact JSON schema:

{json.dumps(schema, indent=2)}

Alert: {ALERT}""",
        }],
    )

    text = response.content[0].text
    print("=== Approach 3: Pydantic Validation ===")

    try:
        # Pydantic validates everything automatically:
        # - Required fields present
        # - Correct types (str, list[str], bool, float)
        # - Value constraints (confidence between 0.0 and 1.0)
        classification = AlertClassification.model_validate_json(text)

        print(f"Valid! Parsed into a typed Python object:")
        print(f"  severity: {classification.severity}")
        print(f"  category: {classification.category}")
        print(f"  client: {classification.affected_client}")
        print(f"  symbols: {classification.affected_symbols}")
        print(f"  root_cause: {classification.root_cause}")
        print(f"  action: {classification.recommended_action}")
        print(f"  escalate: {classification.requires_immediate_escalation}")
        print(f"  confidence: {classification.confidence}")

        # Now you can use it as a regular Python object:
        if classification.requires_immediate_escalation:
            print(f"\n  >>> WOULD PAGE ON-CALL for {classification.affected_client}")

    except ValidationError as e:
        print(f"VALIDATION FAILED:\n{e}")
    except json.JSONDecodeError as e:
        print(f"PARSE FAILED: {e}")

    print()


# ============================================================
# Bonus: Retry pattern for when validation fails
# ============================================================
def approach_3_with_retry():
    """
    In production, even Pydantic-validated calls can fail occasionally.
    This shows a retry pattern: if validation fails, send the error
    back to the model and ask it to fix its response.
    """
    schema = AlertClassification.model_json_schema()

    messages = [{
        "role": "user",
        "content": f"""Classify this alert according to this exact JSON schema:

{json.dumps(schema, indent=2)}

Alert: {ALERT}""",
    }]

    print("=== Approach 3 + Retry Pattern ===")

    for attempt in range(3):
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system="You are a trading alert classifier. Respond with valid JSON only.",
            messages=messages,
        )

        text = response.content[0].text

        try:
            classification = AlertClassification.model_validate_json(text)
            print(f"Success on attempt {attempt + 1}!")
            print(f"  severity={classification.severity}, escalate={classification.requires_immediate_escalation}")
            return classification

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            # Feed the error back to the model so it can self-correct
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"That JSON was invalid. Error: {e}\nPlease fix and return valid JSON only.",
            })

    print("Failed after 3 attempts!")
    return None


# --- Run all approaches ---
if __name__ == "__main__":
    print("=" * 60)
    print("SESSION 2: Structured Outputs")
    print("=" * 60)
    print()

    approach_1_prompt_only()
    approach_2_manual_validation()
    approach_3_pydantic()
    approach_3_with_retry()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Never trust free-form LLM output in production")
    print("  2. Pydantic models = schema + validation + type safety")
    print("  3. Pass the JSON schema to the model so it knows the contract")
    print("  4. Add a retry loop that feeds errors back for self-correction")
    print("  5. Your Pydantic model IS the interface between LLM and your app")
    print("=" * 60)
