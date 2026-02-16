"""
Session 1, Task 2: Basic LLM API Calls
=======================================
Learn the fundamental patterns for calling Claude and GPT-4 APIs.

Run: python 01_basic_api_calls.py
"""

import anthropic
import openai


def call_claude_basic():
    """Basic Claude API call — the simplest possible request."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "What is a token in the context of LLMs? Answer in 2 sentences."}
        ],
    )

    print("=== Claude Basic Call ===")
    print(f"Response: {message.content[0].text}")
    print(f"Model: {message.model}")
    print(f"Input tokens: {message.usage.input_tokens}")
    print(f"Output tokens: {message.usage.output_tokens}")
    print()


def call_claude_with_system_prompt():
    """
    Claude with a system prompt — this is how you set the 'persona' or instructions.

    KEY CONCEPT: The system prompt sets the context/role for the entire conversation.
    It's separate from the user messages and is processed differently by the model.
    """
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=256,
        system="You are a senior trading systems engineer. Explain concepts using trading system analogies.",
        messages=[
            {"role": "user", "content": "What is the difference between temperature 0 and temperature 1 in LLM sampling?"}
        ],
    )

    print("=== Claude with System Prompt ===")
    print(f"Response: {message.content[0].text}")
    print()


def call_claude_multi_turn():
    """
    Multi-turn conversation — the model is stateless, so YOU must send the full history.

    KEY CONCEPT: LLMs don't remember previous calls. Each API call is independent.
    You maintain conversation state by sending all prior messages each time.
    """
    client = anthropic.Anthropic()

    messages = [
        {"role": "user", "content": "What is RAG in AI? One sentence."},
        {"role": "assistant", "content": "RAG (Retrieval Augmented Generation) is a technique that enhances LLM responses by first retrieving relevant documents from a knowledge base and including them in the prompt context."},
        {"role": "user", "content": "What are the main components needed to build one?"},
    ]

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=256,
        messages=messages,
    )

    print("=== Claude Multi-Turn ===")
    print(f"Response: {message.content[0].text}")
    print()


def call_claude_streaming():
    """
    Streaming — tokens arrive one at a time instead of waiting for the full response.

    KEY CONCEPT: Streaming reduces perceived latency. The model starts sending tokens
    as soon as they're generated, rather than waiting for the complete response.
    This is how ChatGPT and Claude.ai show text appearing gradually.
    """
    client = anthropic.Anthropic()

    print("=== Claude Streaming ===")
    print("Response: ", end="", flush=True)

    with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "List 3 ways AI can improve trading system operations. Be brief."}
        ],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n")


def call_claude_with_temperature():
    """
    Temperature experiment — see how temperature affects output randomness.

    KEY CONCEPT:
    - temperature=0: deterministic, always picks the most likely token. Good for factual tasks.
    - temperature=1: more random, explores less likely tokens. Good for creative tasks.
    """
    client = anthropic.Anthropic()

    prompt = "Give me a one-sentence metaphor for what a vector database does."

    print("=== Temperature Comparison ===")
    for temp in [0.0, 0.5, 1.0]:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"  temp={temp}: {message.content[0].text}")

    print()


# --- Run all examples ---
if __name__ == "__main__":
    print("=" * 60)
    print("SESSION 1: Basic LLM API Calls")
    print("=" * 60)
    print()

    call_claude_basic()
    call_claude_with_system_prompt()
    call_claude_multi_turn()
    call_claude_streaming()
    call_claude_with_temperature()

    print("=" * 60)
    print("DONE! Key takeaways:")
    print("  1. Every API call needs: model, max_tokens, messages")
    print("  2. System prompt sets the persona/instructions")
    print("  3. Multi-turn requires sending full conversation history")
    print("  4. Streaming reduces perceived latency")
    print("  5. Temperature controls randomness (0=deterministic, 1=creative)")
    print("=" * 60)
