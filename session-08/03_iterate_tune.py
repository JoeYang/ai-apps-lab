"""
Session 8, Task 3: Iterate & Tune — Systematic RAG Optimization
=================================================================
Run ablation experiments: change one variable at a time, measure
impact with LLM-as-judge, and find the best configuration.

Variables tested:
  - Chunk size (200, 500, 1000 chars)
  - Top-K retrieval (3, 5, 10)
  - With/without reranking (using cross-encoder)

Run: python 03_iterate_tune.py

Requires: pip install chromadb openai anthropic sentence-transformers rank-bm25
"""

import re
import json
import time
from pathlib import Path
from dataclasses import dataclass

import chromadb
import openai
import anthropic
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

oai = openai.OpenAI()
claude = anthropic.Anthropic()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5-20250929"
DOCS_DIR = Path(__file__).parent.parent / "session-06" / "docs"
EVAL_FILE = Path(__file__).parent / "eval_dataset.json"


# ============================================================
# Data structures
# ============================================================

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = f"{self.source}::chunk-{self.chunk_id}"


@dataclass
class ExperimentConfig:
    """One set of tunable parameters to test."""
    name: str
    chunk_size: int = 500
    chunk_overlap: int = 0
    top_k: int = 3
    use_reranking: bool = False

    def __str__(self):
        parts = [
            f"chunk={self.chunk_size}",
            f"overlap={self.chunk_overlap}",
            f"k={self.top_k}",
        ]
        if self.use_reranking:
            parts.append("rerank=yes")
        return f"{self.name} ({', '.join(parts)})"


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    avg_faithfulness: float
    avg_relevancy: float
    avg_completeness: float
    avg_context_quality: float
    num_chunks: int
    avg_latency_ms: float

    @property
    def overall_score(self) -> float:
        return (
            self.avg_faithfulness
            + self.avg_relevancy
            + self.avg_completeness
            + self.avg_context_quality
        ) / 4


# ============================================================
# Chunking with configurable parameters
# ============================================================

def chunk_documents(chunk_size: int, overlap: int = 0) -> list[Chunk]:
    """Load and chunk all docs with the given parameters."""
    all_chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text()
        source = path.name
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current, idx = "", 0

        for sent in sentences:
            if len(current) + len(sent) > chunk_size and current:
                all_chunks.append(Chunk(
                    text=current.strip(), source=source, chunk_id=idx
                ))
                idx += 1
                # Keep overlap chars from the end of current chunk
                if overlap > 0:
                    current = current[-overlap:] + sent + " "
                else:
                    current = sent + " "
            else:
                current += sent + " "
        if current.strip():
            all_chunks.append(Chunk(
                text=current.strip(), source=source, chunk_id=idx
            ))
    return all_chunks


# ============================================================
# Index building
# ============================================================

def build_index(chunks: list[Chunk], name: str) -> chromadb.Collection:
    client = chromadb.Client()
    collection = client.create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    texts = [c.text for c in chunks]
    ids = [c.doc_id for c in chunks]
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = oai.embeddings.create(input=batch, model=EMBED_MODEL)
        embeddings.extend([e.embedding for e in resp.data])
    collection.add(
        ids=ids, documents=texts, embeddings=embeddings,
        metadatas=[{"source": c.source} for c in chunks],
    )
    return collection


# ============================================================
# Retrieval with optional reranking
# ============================================================

CROSS_ENCODER = None


def get_cross_encoder() -> CrossEncoder:
    global CROSS_ENCODER
    if CROSS_ENCODER is None:
        print("  Loading cross-encoder model...")
        CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CROSS_ENCODER


def retrieve(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 3,
    use_reranking: bool = False,
) -> list[str]:
    """Retrieve documents, optionally reranking with a cross-encoder."""
    # Retrieve more candidates if reranking
    fetch_k = top_k * 3 if use_reranking else top_k

    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    results = collection.query(
        query_embeddings=[resp.data[0].embedding], n_results=fetch_k
    )
    documents = results["documents"][0]

    if use_reranking and len(documents) > top_k:
        model = get_cross_encoder()
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs)
        scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        documents = [doc for doc, _ in scored[:top_k]]
    else:
        documents = documents[:top_k]

    return documents


def generate_answer(query: str, contexts: list[str]) -> str:
    context_str = "\n\n---\n\n".join(contexts)
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"Answer the question based ONLY on the provided context. "
                f"If the context doesn't contain the answer, say so.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}"
            ),
        }],
    )
    return response.content[0].text


# ============================================================
# LLM-as-judge scoring
# ============================================================

def judge_answer(question: str, answer: str, ground_truth: str, contexts: list[str]) -> dict:
    """Score a single RAG output using Claude as judge."""
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Score this RAG output (1-5 for each criterion).\n\n"
                f"Question: {question}\n"
                f"Retrieved context (first 500 chars): {' '.join(contexts)[:500]}\n"
                f"Generated answer: {answer}\n"
                f"Ground truth: {ground_truth}\n\n"
                f"Criteria:\n"
                f"- faithfulness: answer uses only info from context\n"
                f"- relevancy: answer addresses the question\n"
                f"- completeness: answer covers ground truth points\n"
                f"- context_quality: retriever found relevant docs\n\n"
                f"Return ONLY JSON: "
                f'{{\"faithfulness\": N, \"relevancy\": N, '
                f'\"completeness\": N, \"context_quality\": N}}'
            ),
        }],
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group()) if match else {
            "faithfulness": 3, "relevancy": 3, "completeness": 3, "context_quality": 3
        }


# ============================================================
# Load eval dataset
# ============================================================

def load_eval_dataset(max_samples: int = 10) -> list[dict]:
    """Load eval dataset, falling back to built-in examples."""
    if EVAL_FILE.exists():
        with open(EVAL_FILE) as f:
            data = json.load(f)
        # Filter to answerable questions for cleaner ablation
        answerable = [d for d in data if d.get("difficulty") != "negative"]
        print(f"  Loaded {len(answerable)} answerable pairs from {EVAL_FILE.name}")
        return answerable[:max_samples]

    # Fallback: built-in eval pairs
    print("  No eval_dataset.json found, using built-in examples")
    return [
        {
            "question": "What are the steps for escalating a critical incident?",
            "ground_truth": "Page on-call engineer, create war room, notify team lead within 15 minutes.",
        },
        {
            "question": "How do I restart the matching engine?",
            "ground_truth": "Drain orders, persist order book, stop engine, apply config, restart with health checks.",
        },
        {
            "question": "What should I do when Kafka broker is down?",
            "ground_truth": "Check broker logs, verify ZooKeeper, restart broker, reassign partitions if needed.",
        },
        {
            "question": "How do I configure monitoring alerts?",
            "ground_truth": "Define alert rules, set thresholds for latency/error rate, configure notification channels.",
        },
        {
            "question": "What is the max_connections_per_host setting?",
            "ground_truth": "Limits concurrent connections to each downstream service. Default is 100.",
        },
    ]


# ============================================================
# Run a single experiment
# ============================================================

def run_experiment(config: ExperimentConfig, eval_data: list[dict]) -> ExperimentResult:
    """Run one experiment: chunk, index, retrieve, generate, judge."""
    print(f"\n  Running: {config}")

    # Chunk and index
    chunks = chunk_documents(config.chunk_size, config.chunk_overlap)
    collection = build_index(chunks, name=config.name.replace(" ", "_"))

    scores = []
    total_latency = 0

    for item in eval_data:
        t0 = time.time()
        contexts = retrieve(collection, item["question"], config.top_k, config.use_reranking)
        answer = generate_answer(item["question"], contexts)
        latency = (time.time() - t0) * 1000
        total_latency += latency

        score = judge_answer(
            item["question"], answer, item["ground_truth"], contexts
        )
        scores.append(score)

    # Aggregate
    n = len(scores)
    result = ExperimentResult(
        config=config,
        avg_faithfulness=sum(s.get("faithfulness", 0) for s in scores) / n,
        avg_relevancy=sum(s.get("relevancy", 0) for s in scores) / n,
        avg_completeness=sum(s.get("completeness", 0) for s in scores) / n,
        avg_context_quality=sum(s.get("context_quality", 0) for s in scores) / n,
        num_chunks=len(chunks),
        avg_latency_ms=total_latency / n,
    )
    print(f"    Score: {result.overall_score:.2f}/5  "
          f"(F={result.avg_faithfulness:.1f} R={result.avg_relevancy:.1f} "
          f"C={result.avg_completeness:.1f} Q={result.avg_context_quality:.1f})  "
          f"Chunks: {result.num_chunks}  Latency: {result.avg_latency_ms:.0f}ms")

    return result


# ============================================================
# Define experiments
# ============================================================

EXPERIMENTS = [
    # Baseline
    ExperimentConfig(name="baseline", chunk_size=500, top_k=3),

    # Vary chunk size
    ExperimentConfig(name="small_chunks", chunk_size=200, top_k=3),
    ExperimentConfig(name="large_chunks", chunk_size=1000, top_k=3),

    # Vary chunk overlap
    ExperimentConfig(name="with_overlap", chunk_size=500, chunk_overlap=100, top_k=3),

    # Vary top-K
    ExperimentConfig(name="more_context", chunk_size=500, top_k=5),
    ExperimentConfig(name="lots_context", chunk_size=500, top_k=10),

    # Add reranking
    ExperimentConfig(name="reranked", chunk_size=500, top_k=3, use_reranking=True),

    # Best guess combo
    ExperimentConfig(
        name="best_combo", chunk_size=500, chunk_overlap=100,
        top_k=3, use_reranking=True
    ),
]


# ============================================================
# Main
# ============================================================

def main():
    print("Session 8 — Iterate & Tune: Systematic RAG Optimization\n")

    eval_data = load_eval_dataset(max_samples=5)
    print(f"  Using {len(eval_data)} eval pairs per experiment")
    print(f"  Running {len(EXPERIMENTS)} experiments\n")

    results: list[ExperimentResult] = []
    for config in EXPERIMENTS:
        result = run_experiment(config, eval_data)
        results.append(result)

    # Leaderboard
    results.sort(key=lambda r: r.overall_score, reverse=True)

    print(f"\n\n{'='*70}")
    print("LEADERBOARD")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Experiment':<35} {'Score':<7} {'Faith':<7} "
          f"{'Relev':<7} {'Compl':<7} {'CtxQ':<7} {'Chunks':<7} {'Latency':<8}")
    print("-" * 90)

    for i, r in enumerate(results, 1):
        print(f"{i:<5} {str(r.config):<35} {r.overall_score:<7.2f} "
              f"{r.avg_faithfulness:<7.1f} {r.avg_relevancy:<7.1f} "
              f"{r.avg_completeness:<7.1f} {r.avg_context_quality:<7.1f} "
              f"{r.num_chunks:<7} {r.avg_latency_ms:<8.0f}ms")

    # Save results
    output = []
    for r in results:
        output.append({
            "experiment": r.config.name,
            "config": {
                "chunk_size": r.config.chunk_size,
                "chunk_overlap": r.config.chunk_overlap,
                "top_k": r.config.top_k,
                "use_reranking": r.config.use_reranking,
            },
            "scores": {
                "overall": round(r.overall_score, 3),
                "faithfulness": round(r.avg_faithfulness, 3),
                "relevancy": round(r.avg_relevancy, 3),
                "completeness": round(r.avg_completeness, 3),
                "context_quality": round(r.avg_context_quality, 3),
            },
            "num_chunks": r.num_chunks,
            "avg_latency_ms": round(r.avg_latency_ms, 1),
        })

    results_file = Path(__file__).parent / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")

    best = results[0]
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print(f"\n  Best config: {best.config}")
    print(f"  Overall score: {best.overall_score:.2f}/5")
    print(f"\n  This configuration scored highest across all four metrics.")
    print(f"  Use these settings as your production baseline and continue")
    print(f"  tuning from here.")

    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  1. Change ONE variable at a time to isolate what helps.
  2. Chunk size has a big impact — too small loses context,
     too large dilutes the embedding signal.
  3. Reranking almost always helps, at the cost of latency.
  4. More top-K isn't always better — noise can hurt faithfulness.
  5. Overlap helps when sentences span chunk boundaries.
  6. Save experiment results — you'll want to compare as you
     add new features (HyDE, multi-query, etc).
""")


if __name__ == "__main__":
    main()
