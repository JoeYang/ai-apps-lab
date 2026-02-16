"""
Session 1, Task 4: Multi-Model Prompt CLI
==========================================
A reusable CLI tool that:
  1. Takes a prompt template with {variables}
  2. Fills in the variables from command-line args
  3. Runs it against one or more models
  4. Displays results side-by-side with token usage and cost

This is the kind of utility you'll use throughout the course
to quickly test prompts across models.

Usage examples:
  # Simple prompt
  python 03_multi_model_cli.py "What is {topic} in 2 sentences?" --topic "RAG"

  # With a system prompt
  python 03_multi_model_cli.py "Explain {concept}" --concept "vector embeddings" \
      --system "You are a trading systems engineer. Use trading analogies."

  # With temperature control
  python 03_multi_model_cli.py "Give me a metaphor for {thing}" --thing "a message queue" \
      --temperature 1.0

  # Compare multiple models (requires OpenAI key)
  python 03_multi_model_cli.py "Summarise {topic}" --topic "RLHF" \
      --models claude-sonnet gpt-4o
"""

import argparse
import re
import sys
import time

import anthropic

# Map friendly names to actual model IDs
MODEL_MAP = {
    "claude-haiku": {"id": "claude-haiku-4-5-20251001", "provider": "anthropic", "cost_in": 0.80, "cost_out": 4.00},
    "claude-sonnet": {"id": "claude-sonnet-4-5-20250929", "provider": "anthropic", "cost_in": 3.00, "cost_out": 15.00},
    "claude-opus": {"id": "claude-opus-4-6", "provider": "anthropic", "cost_in": 15.00, "cost_out": 75.00},
}

# Uncomment if you have an OpenAI key:
# import openai
# MODEL_MAP.update({
#     "gpt-4o": {"id": "gpt-4o", "provider": "openai", "cost_in": 2.50, "cost_out": 10.00},
#     "gpt-4o-mini": {"id": "gpt-4o-mini", "provider": "openai", "cost_in": 0.15, "cost_out": 0.60},
# })


def call_anthropic(model_id, system_prompt, user_prompt, temperature, max_tokens):
    """Call Anthropic's API and return (response_text, input_tokens, output_tokens, elapsed_ms)."""
    client = anthropic.Anthropic()

    kwargs = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if temperature is not None:
        kwargs["temperature"] = temperature

    start = time.time()
    message = client.messages.create(**kwargs)
    elapsed = (time.time() - start) * 1000

    return (
        message.content[0].text,
        message.usage.input_tokens,
        message.usage.output_tokens,
        elapsed,
    )


# Uncomment if you have an OpenAI key:
# def call_openai(model_id, system_prompt, user_prompt, temperature, max_tokens):
#     """Call OpenAI's API and return (response_text, input_tokens, output_tokens, elapsed_ms)."""
#     client = openai.OpenAI()
#
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({"role": "user", "content": user_prompt})
#
#     kwargs = {"model": model_id, "max_tokens": max_tokens, "messages": messages}
#     if temperature is not None:
#         kwargs["temperature"] = temperature
#
#     start = time.time()
#     response = client.chat.completions.create(**kwargs)
#     elapsed = (time.time() - start) * 1000
#
#     return (
#         response.choices[0].message.content,
#         response.usage.prompt_tokens,
#         response.usage.completion_tokens,
#         elapsed,
#     )


def call_model(model_name, system_prompt, user_prompt, temperature, max_tokens):
    """Route to the right provider based on model name."""
    model_info = MODEL_MAP[model_name]

    if model_info["provider"] == "anthropic":
        return call_anthropic(model_info["id"], system_prompt, user_prompt, temperature, max_tokens)
    # elif model_info["provider"] == "openai":
    #     return call_openai(model_info["id"], system_prompt, user_prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {model_info['provider']}")


def estimate_cost(model_name, input_tokens, output_tokens):
    """Estimate cost in USD (per 1M tokens pricing)."""
    info = MODEL_MAP[model_name]
    cost = (input_tokens * info["cost_in"] + output_tokens * info["cost_out"]) / 1_000_000
    return cost


def fill_template(template, variables):
    """Replace {variable} placeholders in the template."""
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", value)

    # Check for unfilled placeholders
    unfilled = re.findall(r"\{(\w+)\}", result)
    if unfilled:
        print(f"Error: unfilled template variables: {unfilled}")
        print(f"Provide them with: {' '.join(f'--{v} VALUE' for v in unfilled)}")
        sys.exit(1)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run a prompt template against one or more LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("template", help="Prompt template with {variables}")
    parser.add_argument("--system", "-s", help="System prompt", default=None)
    parser.add_argument("--models", "-m", nargs="+", default=["claude-sonnet"],
                        choices=list(MODEL_MAP.keys()), help="Models to compare")
    parser.add_argument("--temperature", "-t", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)

    # Parse known args, treat the rest as template variables
    args, unknown = parser.parse_known_args()

    # Parse --variable value pairs from unknown args
    variables = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]  # strip --
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                variables[key] = unknown[i + 1]
                i += 2
            else:
                print(f"Error: --{key} needs a value")
                sys.exit(1)
        else:
            i += 1

    # Fill template
    prompt = fill_template(args.template, variables)

    print("=" * 60)
    print("PROMPT:")
    print(f"  {prompt}")
    if args.system:
        print(f"SYSTEM: {args.system}")
    if args.temperature is not None:
        print(f"TEMPERATURE: {args.temperature}")
    print(f"MODELS: {', '.join(args.models)}")
    print("=" * 60)

    # Run against each model
    for model_name in args.models:
        print(f"\n--- {model_name} ({MODEL_MAP[model_name]['id']}) ---")

        try:
            text, in_tok, out_tok, elapsed = call_model(
                model_name, args.system, prompt, args.temperature, args.max_tokens
            )
            cost = estimate_cost(model_name, in_tok, out_tok)

            print(f"\n{text}\n")
            print(f"  Tokens: {in_tok} in / {out_tok} out")
            print(f"  Latency: {elapsed:.0f}ms")
            print(f"  Est. cost: ${cost:.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    if len(args.models) > 1:
        print("Compare: quality, verbosity, token usage, latency, cost")
    print("=" * 60)


if __name__ == "__main__":
    main()
