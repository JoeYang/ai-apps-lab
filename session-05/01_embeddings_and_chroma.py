"""
Session 5, Task 2-3: Embeddings & Vector Search with Chroma
============================================================
Set up a vector database, index trading documents, and search semantically.

This script:
1. Creates sample trading system documents (incidents, runbooks, configs)
2. Indexes them in Chroma with OpenAI embeddings
3. Demonstrates semantic search vs keyword search
4. Shows how cosine similarity works under the hood

Run: python 01_embeddings_and_chroma.py
"""

import json
import math

import chromadb
import openai

oai = openai.OpenAI()
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims, cheap and fast


# ============================================================
# Sample trading system documents
# ============================================================

DOCUMENTS = [
    {
        "id": "incident-001",
        "text": (
            "FIX Gateway Outage — 2025-11-15. The FIX gateway became unresponsive "
            "at 09:32 EST due to connection pool exhaustion. 12 clients were unable "
            "to submit orders for 23 minutes. Root cause: a market data spike caused "
            "a burst of 50,000 quote updates per second, overwhelming the connection "
            "pool (max 200 connections). Fix: increased pool to 500, added circuit "
            "breaker with 80% threshold. P1 severity."
        ),
        "type": "incident",
        "severity": "P1",
        "system": "fix-gateway",
    },
    {
        "id": "incident-002",
        "text": (
            "Market Data Feed Delay — 2025-12-03. Level 2 market data from NYSE "
            "experienced 500ms latency spikes between 10:15 and 10:20 EST. Cause: "
            "upstream provider (Reuters) had a network issue in their NJ datacenter. "
            "Impact: some limit orders were priced on stale data. 3 clients reported "
            "worse-than-expected fills. P3 severity. No action needed on our side."
        ),
        "type": "incident",
        "severity": "P3",
        "system": "market-data",
    },
    {
        "id": "incident-003",
        "text": (
            "Order Rejection Spike — 2025-12-20. Between 14:00 and 14:02 EST, "
            "order rejection rate spiked to 45% (normal: <2%). Root cause: a config "
            "deployment pushed incorrect risk limits for 3 client accounts. The new "
            "max position size was set to $100 instead of $100,000. Detected by "
            "the anomaly alerting system. Rolled back in 2 minutes. P4 severity."
        ),
        "type": "incident",
        "severity": "P4",
        "system": "risk-engine",
    },
    {
        "id": "incident-004",
        "text": (
            "Sequencer Failover Event — 2026-01-08. Primary sequencer node lost "
            "heartbeat at 11:45 EST. Automatic failover to secondary completed in "
            "340μs. No orders were lost. Root cause: a kernel page fault triggered "
            "by a memory-mapped file growing beyond the pre-allocated region. "
            "Prevention: increased mmap pre-allocation to 2x expected daily volume. "
            "P2 severity due to failover, but no client impact."
        ),
        "type": "incident",
        "severity": "P2",
        "system": "sequencer",
    },
    {
        "id": "runbook-001",
        "text": (
            "Runbook: FIX Gateway Reconnection. When a FIX session disconnects: "
            "1) Check the gateway logs for disconnect reason (network, heartbeat "
            "timeout, or sequence number mismatch). 2) If sequence mismatch, request "
            "a resend from the counterparty. 3) If network issue, verify connectivity "
            "with ping and traceroute to the exchange endpoint. 4) If heartbeat "
            "timeout, check system load — high CPU can delay heartbeats. 5) Restart "
            "the FIX session with ResetOnLogon=Y if sequence recovery fails."
        ),
        "type": "runbook",
        "severity": "N/A",
        "system": "fix-gateway",
    },
    {
        "id": "runbook-002",
        "text": (
            "Runbook: High Latency Investigation. When p99 latency exceeds threshold: "
            "1) Check if the spike is system-wide or isolated to specific clients. "
            "2) Run `perf top` to identify hot functions. 3) Check for GC pauses in "
            "Java components (use -verbose:gc logs). 4) Verify network latency with "
            "timestamped packet captures. 5) Check if the issue correlates with "
            "market volatility (more orders = more load). 6) If isolated to one "
            "client, check their order patterns for unusual volume."
        ),
        "type": "runbook",
        "severity": "N/A",
        "system": "general",
    },
    {
        "id": "runbook-003",
        "text": (
            "Runbook: Market Data Feed Recovery. When market data stops updating: "
            "1) Check feed handler process status. 2) Verify multicast group "
            "membership with `netstat -gn`. 3) Check for packet loss on the "
            "market data NIC with `ethtool -S`. 4) If packets are arriving but "
            "not decoded, check the schema version matches the exchange's current "
            "format. 5) Failover to backup feed if primary is down. 6) Notify "
            "clients if stale data was served."
        ),
        "type": "runbook",
        "severity": "N/A",
        "system": "market-data",
    },
    {
        "id": "config-001",
        "text": (
            "Risk Engine Configuration. Default risk limits: max position size "
            "$10M per symbol, max daily loss $500K per client, max order rate "
            "100 orders/second per client. Kill switch triggers at $1M daily loss. "
            "Pre-trade risk checks: position limit, daily loss, order rate, "
            "fat finger (order size > 10x average). All limits configurable per "
            "client via the risk admin portal."
        ),
        "type": "config",
        "severity": "N/A",
        "system": "risk-engine",
    },
    {
        "id": "config-002",
        "text": (
            "Network Architecture. Trading network is physically isolated from "
            "corporate network. Market data feeds arrive via dedicated 10Gbps "
            "links from exchanges. Order routing uses kernel bypass (DPDK) for "
            "sub-microsecond latency. Two redundant paths to each exchange with "
            "automatic failover. Monitoring uses out-of-band IPMI for hardware "
            "health. All inter-component communication uses shared memory (no TCP "
            "on the hot path)."
        ),
        "type": "config",
        "severity": "N/A",
        "system": "network",
    },
    {
        "id": "postmortem-001",
        "text": (
            "Postmortem: Client HEDGE_FUND_A Overcharged — 2026-01-15. Client was "
            "charged $12,400 in fees instead of $1,240 due to a decimal point error "
            "in the fee calculation module. The bug was introduced in release v2.34 "
            "when the fee multiplier was changed from 0.001 to 0.01 without updating "
            "the downstream calculation. Detected by the client 3 days later. "
            "Refunded $11,160. Added automated fee reconciliation check to CI pipeline."
        ),
        "type": "postmortem",
        "severity": "P2",
        "system": "billing",
    },
]


# ============================================================
# Manual cosine similarity (to understand the math)
# ============================================================

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    cos(θ) = (A · B) / (||A|| × ||B||)

    - 1.0  = identical direction (same meaning)
    - 0.0  = orthogonal (unrelated)
    - -1.0 = opposite direction (opposite meaning)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# Embedding helper
# ============================================================

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI API."""
    response = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ============================================================
# Main demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SESSION 5: Embeddings & Vector Search with Chroma")
    print("=" * 60)

    # ----------------------------------------------------------
    # Part 1: Understanding embeddings
    # ----------------------------------------------------------
    print("\n--- Part 1: What do embeddings look like? ---\n")

    sample_texts = [
        "FIX gateway connection pool exhaustion",
        "FIX session disconnected due to network failure",
        "The weather is sunny and warm today",
    ]

    embeddings = get_embeddings(sample_texts)

    for text, emb in zip(sample_texts, embeddings):
        print(f"  \"{text}\"")
        print(f"    → {len(emb)} dimensions, first 5: [{', '.join(f'{x:.4f}' for x in emb[:5])}...]")

    print(f"\n  Similarity (FIX pool ↔ FIX network):  {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"  Similarity (FIX pool ↔ weather):       {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"  Similarity (FIX network ↔ weather):    {cosine_similarity(embeddings[1], embeddings[2]):.4f}")
    print("\n  → Related texts have high similarity; unrelated texts have low similarity")

    # ----------------------------------------------------------
    # Part 2: Index documents in Chroma
    # ----------------------------------------------------------
    print("\n--- Part 2: Indexing documents in Chroma ---\n")

    # Create a Chroma client (in-memory for demo, persistent for production)
    chroma = chromadb.Client()  # In-memory

    # Create a collection (like a table in a database)
    collection = chroma.create_collection(
        name="trading_docs",
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )

    # Index all documents
    doc_embeddings = get_embeddings([doc["text"] for doc in DOCUMENTS])

    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        embeddings=doc_embeddings,
        metadatas=[{
            "type": doc["type"],
            "severity": doc["severity"],
            "system": doc["system"],
        } for doc in DOCUMENTS],
    )

    print(f"  Indexed {collection.count()} documents in Chroma")
    print(f"  Document types: {set(doc['type'] for doc in DOCUMENTS)}")

    # ----------------------------------------------------------
    # Part 3: Semantic search
    # ----------------------------------------------------------
    print("\n--- Part 3: Semantic Search ---\n")

    queries = [
        "FIX gateway went down and clients couldn't trade",
        "how to debug slow order execution",
        "what are the risk limits for trading",
        "client was billed incorrectly",
        "network setup for low latency trading",
    ]

    for query in queries:
        results = collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        print(f"  Query: \"{query}\"")
        for i in range(3):
            doc_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - distance  # Chroma returns distance, not similarity
            doc_type = results["metadatas"][0][i]["type"]
            doc_text = results["documents"][0][i][:80]
            print(f"    {i+1}. [{doc_type}] {doc_id} (similarity: {similarity:.3f})")
            print(f"       {doc_text}...")
        print()

    # ----------------------------------------------------------
    # Part 4: Semantic vs keyword — where embeddings shine
    # ----------------------------------------------------------
    print("--- Part 4: Semantic vs Keyword Search ---\n")

    # These queries use DIFFERENT WORDS but the same MEANING
    semantic_queries = [
        ("orders getting rejected a lot", "Should find incident-003 (rejection spike)"),
        ("system crashed and switched to backup", "Should find incident-004 (sequencer failover)"),
        ("customer overcharged on fees", "Should find postmortem-001 (fee error)"),
    ]

    for query, expected in semantic_queries:
        results = collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=1,
        )
        top_id = results["ids"][0][0]
        distance = results["distances"][0][0]
        print(f"  Query: \"{query}\"")
        print(f"  Expected: {expected}")
        print(f"  Got: {top_id} (similarity: {1 - distance:.3f})")
        print(f"  {'MATCH' if expected.split('(')[0].strip().endswith(top_id) or top_id in expected else 'Check result'}")
        print()

    # ----------------------------------------------------------
    # Part 5: Metadata filtering
    # ----------------------------------------------------------
    print("--- Part 5: Metadata Filtering ---\n")

    # Search only within incidents
    results = collection.query(
        query_embeddings=get_embeddings(["connection issues"]),
        n_results=3,
        where={"type": "incident"},
        include=["metadatas", "distances"],
    )
    print("  Query: \"connection issues\" (filtered: incidents only)")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        severity = results["metadatas"][0][i]["severity"]
        similarity = 1 - results["distances"][0][i]
        print(f"    {i+1}. {doc_id} [{severity}] (similarity: {similarity:.3f})")

    print()

    # Search only within a specific system
    results = collection.query(
        query_embeddings=get_embeddings(["troubleshooting steps"]),
        n_results=3,
        where={"type": "runbook"},
        include=["metadatas", "distances"],
    )
    print("  Query: \"troubleshooting steps\" (filtered: runbooks only)")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        system = results["metadatas"][0][i]["system"]
        similarity = 1 - results["distances"][0][i]
        print(f"    {i+1}. {doc_id} [{system}] (similarity: {similarity:.3f})")

    print()
    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Embeddings convert text → vectors; similar meaning = close vectors")
    print("  2. Cosine similarity measures how 'close' two vectors are (0 to 1)")
    print("  3. Chroma stores vectors + metadata and searches by similarity")
    print("  4. Semantic search finds results even when WORDS are different")
    print("     (\"system crashed\" matches \"failover event\")")
    print("  5. Metadata filtering narrows results (type=incident, system=fix-gateway)")
    print("  6. This is the foundation of RAG — retrieve relevant docs, then generate")
    print("=" * 60)
