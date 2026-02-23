"""
Session 11, Task 3 (continued): Framework comparison summary.

Side-by-side comparison of the three approaches we've used across Sessions 9-11.
"""

comparison = """
╔══════════════════════════════════════════════════════════════════════════╗
║           MULTI-AGENT FRAMEWORK COMPARISON                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  APPROACH         │ Simple Loop    │ LangGraph       │ OpenAI Agents SDK ║
║  (Session)        │ (Session 9)    │ (Session 10-11) │ (Session 11)      ║
║  ─────────────────┼────────────────┼─────────────────┼─────────────────  ║
║  Control flow     │ while loop     │ State graph     │ Handoff-based     ║
║  Who routes?      │ Your code      │ Edges/functions │ LLM decides       ║
║  Agent comm       │ N/A (1 agent)  │ Shared state    │ Handoff messages  ║
║  State mgmt       │ List of msgs   │ TypedDict       │ Conversation obj  ║
║  Persistence      │ DIY            │ Checkpointer    │ Built-in          ║
║  Human-in-loop    │ DIY            │ interrupt_before│ Guardrails        ║
║  Multi-agent      │ Hard           │ Native          │ Native            ║
║  Async            │ DIY            │ Supported       │ Native (asyncio)  ║
║  Tracing          │ Manual logging │ LangSmith       │ Built-in traces   ║
║  Model support    │ Any            │ Any (LangChain) │ OpenAI only*      ║
║  Learning curve   │ Low            │ Medium          │ Low-Medium        ║
║  Vendor lock-in   │ None           │ LangChain       │ OpenAI            ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  KEY DIFFERENCES:                                                        ║
║                                                                          ║
║  1. ROUTING CONTROL                                                      ║
║     LangGraph: YOU define edges → deterministic, predictable             ║
║     OpenAI SDK: LLM decides handoffs → flexible, but unpredictable       ║
║     (We saw this: the Researcher didn't hand off to Writer!)             ║
║                                                                          ║
║  2. STATE MODEL                                                          ║
║     LangGraph: Rich typed state (research, draft, review fields)         ║
║     OpenAI SDK: Conversation-based (messages only)                       ║
║     → LangGraph is better for complex workflows with structured data     ║
║                                                                          ║
║  3. AGENT DEFINITION                                                     ║
║     LangGraph: Agents are nodes (functions). You wire them manually.     ║
║     OpenAI SDK: Agents are objects with instructions + tools + handoffs.  ║
║     → OpenAI SDK is more declarative and quicker to set up               ║
║                                                                          ║
║  4. FAILURE MODE                                                         ║
║     LangGraph: If routing is wrong, you fix the edge logic               ║
║     OpenAI SDK: If LLM skips a handoff, you adjust instructions          ║
║     → LangGraph failures are more debuggable                             ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  WHEN TO USE WHAT:                                                       ║
║                                                                          ║
║  Simple loop:     Prototyping, single agent, full control                ║
║  LangGraph:       Complex workflows, structured state, deterministic     ║
║                   routing, persistence, production systems               ║
║  OpenAI SDK:      Quick multi-agent prototypes, conversational handoffs, ║
║                   OpenAI-only projects, simpler coordination needs       ║
║                                                                          ║
║  * OpenAI Agents SDK technically supports other providers via model      ║
║    configuration, but is optimised for OpenAI models.                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

print(comparison)
