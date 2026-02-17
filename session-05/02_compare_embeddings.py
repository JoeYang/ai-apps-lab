"""
Session 5, Task 4: Compare Embedding Models
=============================================
Compare OpenAI's embedding models on the same trading system queries.

We test:
1. text-embedding-3-small (1536 dims, cheap)
2. text-embedding-3-large (3072 dims, best quality)
3. text-embedding-3-large truncated to 1024 dims (Matryoshka trick)

Run: python 02_compare_embeddings.py
"""

import math
import time

import openai

oai = openai.OpenAI()

MODELS = [
    {"name": "text-embedding-3-small", "dimensions": None, "label": "3-small (1536d)"},
    {"name": "text-embedding-3-large", "dimensions": None, "label": "3-large (3072d)"},
    {"name": "text-embedding-3-large", "dimensions": 1024, "label": "3-large→1024d (Matryoshka)"},
    {"name": "text-embedding-3-large", "dimensions": 256, "label": "3-large→256d (Matryoshka)"},
]

# Documents to index (small set for comparison)
DOCUMENTS = {
    "fix-outage": (
        "FIX Gateway Outage — The FIX gateway became unresponsive due to "
        "connection pool exhaustion. 12 clients were unable to submit orders "
        "for 23 minutes. Root cause: market data spike overwhelmed the pool."
    ),
    "sequencer-failover": (
        "Sequencer Failover Event — Primary sequencer node lost heartbeat. "
        "Automatic failover to secondary completed in 340μs. No orders lost. "
        "Root cause: kernel page fault from memory-mapped file overflow."
    ),
    "latency-runbook": (
        "Runbook: High Latency Investigation. Check if spike is system-wide "
        "or client-specific. Run perf top for hot functions. Check GC pauses "
        "in Java components. Verify network latency with packet captures."
    ),
    "risk-config": (
        "Risk Engine Configuration. Max position size $10M per symbol, max "
        "daily loss $500K per client, max order rate 100 orders/second. "
        "Kill switch triggers at $1M daily loss."
    ),
    "fee-error": (
        "Client HEDGE_FUND_A was charged $12,400 in fees instead of $1,240 "
        "due to a decimal point error in the fee calculation module. Bug "
        "introduced in release v2.34."
    ),
}

# Queries designed to test semantic understanding
QUERIES = [
    {
        "text": "clients couldn't place orders because the system was down",
        "expected": "fix-outage",
        "why": "Synonyms: 'couldn't place orders' = 'unable to submit orders', 'system down' = 'unresponsive'",
    },
    {
        "text": "the backup system took over automatically",
        "expected": "sequencer-failover",
        "why": "Synonyms: 'backup took over' = 'failover to secondary'",
    },
    {
        "text": "how to investigate slow performance",
        "expected": "latency-runbook",
        "why": "Synonyms: 'slow performance' = 'high latency'",
    },
    {
        "text": "trading limits and position controls",
        "expected": "risk-config",
        "why": "Synonyms: 'trading limits' = 'risk limits', 'position controls' = 'position size'",
    },
    {
        "text": "billing mistake with wrong amount charged",
        "expected": "fee-error",
        "why": "Synonyms: 'billing mistake' = 'fee error', 'wrong amount' = 'decimal point error'",
    },
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def get_embeddings(texts: list[str], model: str, dimensions: int | None = None) -> list[list[float]]:
    kwargs = {"model": model, "input": texts}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    response = oai.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 5: Compare Embedding Models")
    print("=" * 70)

    doc_ids = list(DOCUMENTS.keys())
    doc_texts = list(DOCUMENTS.values())
    query_texts = [q["text"] for q in QUERIES]

    results_table = {}  # model_label → list of (query_idx, correct, similarity)

    for model_config in MODELS:
        label = model_config["label"]
        print(f"\n--- {label} ---")

        start = time.time()

        # Embed documents
        doc_embeds = get_embeddings(
            doc_texts, model_config["name"], model_config["dimensions"]
        )

        # Embed queries
        query_embeds = get_embeddings(
            query_texts, model_config["name"], model_config["dimensions"]
        )

        elapsed_ms = int((time.time() - start) * 1000)
        dims = len(doc_embeds[0])
        print(f"  Dimensions: {dims}, Embed time: {elapsed_ms}ms")

        model_results = []
        for i, (query, q_embed) in enumerate(zip(QUERIES, query_embeds)):
            # Compute similarity to all documents
            similarities = {}
            for doc_id, d_embed in zip(doc_ids, doc_embeds):
                similarities[doc_id] = cosine_similarity(q_embed, d_embed)

            # Rank by similarity
            ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_id = ranked[0][0]
            top_sim = ranked[0][1]
            correct = top_id == query["expected"]

            model_results.append((i, correct, top_sim))

            status = "OK" if correct else "WRONG"
            print(f"  [{status}] \"{query['text'][:50]}...\"")
            print(f"         Top: {top_id} ({top_sim:.4f}), Expected: {query['expected']}")
            if not correct:
                expected_sim = similarities[query["expected"]]
                print(f"         Expected was ranked with sim: {expected_sim:.4f}")

        results_table[label] = model_results

    # ----------------------------------------------------------
    # Summary comparison table
    # ----------------------------------------------------------
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Model':<30} {'Accuracy':<12} {'Avg Similarity':<15} {'Dims'}")
    print(f"  {'-'*30} {'-'*12} {'-'*15} {'-'*6}")

    for model_config in MODELS:
        label = model_config["label"]
        model_results = results_table[label]
        accuracy = sum(1 for _, c, _ in model_results if c) / len(model_results)
        avg_sim = sum(s for _, _, s in model_results) / len(model_results)
        dims = model_config["dimensions"] or {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}[model_config["name"]]
        print(f"  {label:<30} {accuracy:>8.0%}     {avg_sim:>10.4f}      {dims}")

    print()
    print("KEY TAKEAWAYS:")
    print("  1. All models should get 5/5 on these queries — they're not adversarial")
    print("  2. 3-large has higher similarity scores (more confident matches)")
    print("  3. Matryoshka trick: 3-large→1024d retains most quality at 1/3 the storage")
    print("  4. 3-large→256d may show quality degradation — that's the tradeoff")
    print("  5. For production: start with 3-small, upgrade to 3-large only if retrieval")
    print("     quality is insufficient. Measure with an eval, not gut feeling.")
    print("  6. Cost comparison:")
    print("     - 3-small: $0.02 / 1M tokens")
    print("     - 3-large: $0.13 / 1M tokens (6.5x more expensive)")
    print("     - Open source (local): $0 but uses your CPU/GPU")
    print("=" * 70)
