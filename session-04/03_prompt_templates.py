"""
Session 4, Task 3: Prompt Template Library
==========================================
Reusable, production-grade prompt templates for common tasks.

Each template:
1. Has clear structure (role, task, constraints, output format)
2. Uses XML tags to delimit sections (Claude's preferred format)
3. Returns structured output (parseable)
4. Is parameterised — fill in the blanks, not rewrite from scratch

Run: python 03_prompt_templates.py
"""

import json
from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel, Field

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5-20250929"


# ============================================================
# Template system
# ============================================================

@dataclass
class PromptTemplate:
    """A reusable prompt template with named parameters."""

    name: str
    description: str
    system: str
    user: str
    parameters: list[str] = field(default_factory=list)

    def render(self, **kwargs) -> tuple[str, str]:
        """Render the template with provided parameters.

        Returns (system_prompt, user_message).
        """
        missing = [p for p in self.parameters if p not in kwargs]
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        return self.system.format(**kwargs), self.user.format(**kwargs)


# ============================================================
# Template 1: CODE REVIEW
# ============================================================

code_review = PromptTemplate(
    name="code_review",
    description="Review code for bugs, security issues, and improvements",
    parameters=["language", "code", "context"],
    system="""You are a senior software engineer performing a code review.

## Your review MUST cover:
1. **Bugs** — logic errors, off-by-one, null/undefined issues
2. **Security** — injection, leaks, authentication/authorisation flaws
3. **Performance** — unnecessary allocations, O(n²) where O(n) is possible
4. **Readability** — naming, structure, comments (only where non-obvious)

## Rules:
- Be specific: reference line numbers or code snippets
- Be constructive: explain WHY something is an issue and suggest a fix
- Don't nitpick style unless it hurts readability
- If the code is good, say so — don't invent problems

## Output format:
Respond with a JSON object:
{{
  "summary": "One-line overall assessment",
  "severity": "clean | minor | major | critical",
  "issues": [
    {{
      "type": "bug | security | performance | readability",
      "severity": "info | warning | error",
      "location": "line or function name",
      "description": "What's wrong",
      "suggestion": "How to fix it"
    }}
  ],
  "positive": ["Things done well"]
}}""",
    user="""Review this {language} code.

<context>{context}</context>

<code>
{code}
</code>"""
)


# ============================================================
# Template 2: SUMMARISATION
# ============================================================

summarisation = PromptTemplate(
    name="summarisation",
    description="Summarise a document with configurable detail level",
    parameters=["document", "audience", "max_bullets"],
    system="""You are an expert at distilling complex information into clear summaries.

## Rules:
- Focus on WHAT matters and WHY, not just WHAT happened
- Tailor language to the audience
- Use concrete numbers and facts, not vague statements
- If the document contains action items, list them separately

## Output format:
Respond with a JSON object:
{{
  "title": "Brief descriptive title",
  "one_liner": "Single sentence summary",
  "key_points": ["Up to {max_bullets} bullet points"],
  "action_items": ["Any actions required — empty list if none"],
  "audience_note": "One line on why this matters for {audience}"
}}""",
    user="""Summarise this document for {audience}. Maximum {max_bullets} key points.

<document>
{document}
</document>"""
)


# ============================================================
# Template 3: DATA EXTRACTION
# ============================================================

data_extraction = PromptTemplate(
    name="data_extraction",
    description="Extract structured data from unstructured text",
    parameters=["text", "schema_description", "schema_json"],
    system="""You are a precise data extraction engine.

## Rules:
- Extract ONLY what is explicitly stated in the text
- If a field is not mentioned, use null — NEVER guess or infer
- Preserve exact numbers, dates, and names as written
- If the text is ambiguous, extract the most likely interpretation and flag it

## Output format:
Respond with a JSON object matching this schema:
{schema_json}

If you're uncertain about any field, add an "_uncertain" list with the field names.""",
    user="""Extract data from this text according to the schema: {schema_description}

<text>
{text}
</text>"""
)


# ============================================================
# Template 4: INCIDENT ANALYSIS (trading-specific)
# ============================================================

incident_analysis = PromptTemplate(
    name="incident_analysis",
    description="Analyse a production incident from logs and alerts",
    parameters=["incident_description", "logs", "system_context"],
    system="""You are a senior SRE / production support engineer for a trading system.

## Your analysis MUST include:
1. **Root cause** — what went wrong and why
2. **Impact** — what was affected, for how long, severity
3. **Timeline** — sequence of events from first signal to resolution
4. **Fix** — immediate remediation and longer-term prevention

## Rules:
- Be specific about timestamps, error codes, and affected components
- Distinguish between confirmed facts and hypotheses
- Prioritise business impact (missed trades, wrong prices, client impact)
- Suggest monitoring improvements to catch this earlier next time

## Output format:
Respond with a JSON object:
{{
  "root_cause": "Clear one-line root cause",
  "severity": "P1 | P2 | P3 | P4",
  "impact": {{
    "description": "What was affected",
    "duration_minutes": 0,
    "clients_affected": 0
  }},
  "timeline": [
    {{"time": "HH:MM", "event": "What happened"}}
  ],
  "immediate_fix": "What was done to resolve it",
  "prevention": ["Long-term fixes to prevent recurrence"],
  "monitoring_gaps": ["What should we alert on that we didn't"]
}}""",
    user="""Analyse this production incident.

<system_context>{system_context}</system_context>

<incident>{incident_description}</incident>

<logs>
{logs}
</logs>"""
)


# ============================================================
# Template registry
# ============================================================

TEMPLATES = {
    "code_review": code_review,
    "summarisation": summarisation,
    "data_extraction": data_extraction,
    "incident_analysis": incident_analysis,
}


def run_template(template_name: str, **kwargs) -> dict:
    """Run a template and return the parsed JSON response."""
    template = TEMPLATES[template_name]
    system_prompt, user_message = template.render(**kwargs)

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text

    # Extract JSON from the response (handle markdown code blocks)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    return json.loads(raw)


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SESSION 4: Prompt Template Library")
    print("=" * 60)
    print(f"Templates available: {list(TEMPLATES.keys())}")
    print()

    # --- Demo 1: Code Review ---
    print("-" * 60)
    print("DEMO 1: Code Review Template")
    print("-" * 60)

    sample_code = """\
def process_order(order_data, db):
    client_id = order_data["client_id"]
    amount = order_data["amount"]
    symbol = order_data["symbol"]

    # Check balance
    balance = db.execute(f"SELECT balance FROM accounts WHERE client_id = '{client_id}'")
    if balance[0] > amount:
        db.execute(f"INSERT INTO orders VALUES ('{client_id}', '{symbol}', {amount})")
        db.execute(f"UPDATE accounts SET balance = balance - {amount} WHERE client_id = '{client_id}'")
        return {"status": "filled", "order_id": hash(str(order_data))}
    return {"status": "rejected", "reason": "insufficient balance"}
"""

    result = run_template(
        "code_review",
        language="Python",
        code=sample_code,
        context="Order processing function for a trading system. Called from the REST API handler.",
    )
    print(f"Summary: {result['summary']}")
    print(f"Severity: {result['severity']}")
    for issue in result.get("issues", []):
        print(f"  [{issue['severity']}] {issue['type']}: {issue['description']}")
        print(f"         Fix: {issue['suggestion']}")
    for pos in result.get("positive", []):
        print(f"  [+] {pos}")

    # --- Demo 2: Summarisation ---
    print()
    print("-" * 60)
    print("DEMO 2: Summarisation Template")
    print("-" * 60)

    report = """
    Q4 2025 Trading Operations Report — Acme Trading Corp

    Executive Summary: Q4 saw record trading volumes with 1.2M trades executed,
    up 34% from Q3. Revenue hit $45M with a 12% net margin. System uptime was
    99.97%, with only one P1 incident (the FIX gateway outage on Nov 15 that
    lasted 23 minutes and affected 12 clients).

    Key metrics:
    - Average latency: 450μs (down from 520μs in Q3 after the kernel bypass upgrade)
    - Fill rate: 98.2% (target: 97%)
    - Client satisfaction: 94% (up from 91%)
    - New clients onboarded: 8 (including two hedge funds)

    Incidents:
    - Nov 15: FIX gateway outage, 23 min, P1. Root cause: connection pool exhaustion
      under load spike. Fix: increased pool size, added circuit breaker.
    - Dec 3: Market data feed delay, 5 min, P3. Root cause: upstream provider issue.
    - Dec 20: Order rejection spike, 2 min, P4. Root cause: config deployment error.

    Planned for Q1 2026:
    - Migrate to FPGA-accelerated market data processing
    - Launch new dark pool integration
    - Hire 2 additional SREs for 24/7 coverage
    """

    result = run_template(
        "summarisation",
        document=report,
        audience="engineering team leads",
        max_bullets="5",
    )
    print(f"Title: {result['title']}")
    print(f"One-liner: {result['one_liner']}")
    print("Key points:")
    for point in result["key_points"]:
        print(f"  • {point}")
    if result.get("action_items"):
        print("Action items:")
        for item in result["action_items"]:
            print(f"  → {item}")

    # --- Demo 3: Data Extraction ---
    print()
    print("-" * 60)
    print("DEMO 3: Data Extraction Template")
    print("-" * 60)

    trade_email = """
    Hi Bob,

    Just confirming the block trade we discussed: we'd like to buy 50,000 shares
    of NVDA at limit price $142.50. Account is HEDGE_FUND_A, executing through
    the dark pool. Needs to be done by market close today (4:00 PM EST).

    If we can't fill the full block, partial fills are acceptable down to 25,000
    minimum. Settlement is T+1.

    Thanks,
    Sarah Chen
    Portfolio Manager, Apex Capital
    """

    schema = {
        "action": "buy | sell",
        "symbol": "str",
        "quantity": "int",
        "price_type": "market | limit",
        "limit_price": "float | null",
        "account": "str",
        "venue": "str",
        "deadline": "str",
        "min_fill_quantity": "int | null",
        "settlement": "str",
        "sender_name": "str",
        "sender_firm": "str",
    }

    result = run_template(
        "data_extraction",
        text=trade_email,
        schema_description="Extract trade order details from this email",
        schema_json=json.dumps(schema, indent=2),
    )
    print("Extracted trade order:")
    for key, value in result.items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")

    print()
    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Templates encode BEST PRACTICES — you don't reinvent the wheel each time")
    print("  2. Structured output (JSON) makes templates composable with other code")
    print("  3. XML tags (<code>, <document>) create clear boundaries for the LLM")
    print("  4. Each template has clear RULES and OUTPUT FORMAT — reduces hallucination")
    print("  5. Domain-specific templates (incident analysis) capture team knowledge")
    print("=" * 60)
