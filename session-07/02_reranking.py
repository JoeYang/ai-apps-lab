"""
Session 7, Task 2: Reranking with Cross-Encoders
==================================================
Add a reranking step on top of first-stage retrieval (hybrid search
from Task 1) and measure the improvement.

Two approaches:
  1. Cohere Rerank API — hosted, fast, high quality
  2. Local cross-encoder — sentence-transformers, free, runs on CPU

Run: python 02_reranking.py

Requires: pip install chromadb openai rank-bm25 cohere sentence-transformers
"""

import re
import time
from pathlib import Path
from dataclasses import dataclass

import chromadb
import openai
import cohere
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

oai = openai.OpenAI()
co = cohere.Client()  # uses COHERE_API_KEY env var

EMBED_MODEL = "text-embedding-3-small"
DOCS_DIR = Path(__file__).parent.parent / "session-06" / "docs"


# ============================================================
# Reuse chunking and indexing from 01_hybrid_search
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


def sentence_chunk(text: str, source: str, max_chars: int = 500) -> list[Chunk]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, idx = [], "", 0
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(Chunk(text=current.strip(), source=source, chunk_id=idx))
            idx += 1
            current = ""
        current += sent + " "
    if current.strip():
        chunks.append(Chunk(text=current.strip(), source=source, chunk_id=idx))
    return chunks


def load_and_chunk() -> list[Chunk]:
    all_chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text()
        chunks = sentence_chunk(text, source=path.name)
        all_chunks.extend(chunks)
    print(f"Loaded {len(all_chunks)} chunks from {len(list(DOCS_DIR.glob('*.md')))} documents")
    return all_chunks


def build_vector_index(chunks: list[Chunk]) -> chromadb.Collection:
    client = chromadb.Client()
    collection = client.create_collection(
        name="rerank_demo", metadata={"hnsw:space": "cosine"}
    )
    texts = [c.text for c in chunks]
    ids = [c.doc_id for c in chunks]
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = oai.embeddings.create(input=batch, model=EMBED_MODEL)
        all_embeddings.extend([e.embedding for e in resp.data])
    collection.add(
        ids=ids, documents=texts, embeddings=all_embeddings,
        metadatas=[{"source": c.source} for c in chunks],
    )
    return collection


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.doc_ids = [c.doc_id for c in chunks]
        self.tokenized = [re.split(r'\W+', c.text.lower()) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        tokens = re.split(r'\W+', query.lower())
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def embed_query(query: str) -> list[float]:
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    return resp.data[0].embedding


# ============================================================
# First-stage retrieval: hybrid search with RRF
# ============================================================

def hybrid_search(
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    query: str,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[str]:
    """Return top_k doc IDs from hybrid search (vector + BM25 with RRF)."""
    # Vector search
    vec_results = collection.query(
        query_embeddings=[embed_query(query)], n_results=top_k
    )
    vec_ids = vec_results["ids"][0]

    # BM25 search
    bm25_results = bm25_index.search(query, top_k=top_k)
    bm25_ids = [doc_id for doc_id, _ in bm25_results]

    # RRF fusion
    scores: dict[str, float] = {}
    for ranking in [vec_ids, bm25_ids]:
        for rank_pos, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank_pos)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_k]]


def get_chunk_text(chunks: list[Chunk], doc_id: str) -> str:
    for c in chunks:
        if c.doc_id == doc_id:
            return c.text
    return "(not found)"


# ============================================================
# Reranker 1: Cohere Rerank API
# ============================================================

def cohere_rerank(
    query: str,
    doc_ids: list[str],
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Rerank candidates using Cohere's rerank API.

    Takes the top-N from first-stage retrieval and reorders them
    based on a cross-encoder relevance score.
    """
    documents = [get_chunk_text(chunks, doc_id) for doc_id in doc_ids]

    response = co.rerank(
        query=query,
        documents=documents,
        model="rerank-v3.5",
        top_n=top_k,
    )

    reranked = []
    for result in response.results:
        reranked.append((doc_ids[result.index], result.relevance_score))
    return reranked


# ============================================================
# Reranker 2: Local cross-encoder (sentence-transformers)
# ============================================================

# Load once — this downloads ~80MB the first time
CROSS_ENCODER = None


def get_cross_encoder() -> CrossEncoder:
    global CROSS_ENCODER
    if CROSS_ENCODER is None:
        print("Loading cross-encoder model (first time may download ~80MB)...")
        CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CROSS_ENCODER


def local_rerank(
    query: str,
    doc_ids: list[str],
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Rerank candidates using a local cross-encoder model.

    The model takes (query, document) pairs and scores their relevance.
    Much slower than Cohere API but free and runs locally.
    """
    model = get_cross_encoder()
    documents = [get_chunk_text(chunks, doc_id) for doc_id in doc_ids]

    # Cross-encoder expects list of [query, document] pairs
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)

    # Sort by score descending
    scored = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, float(score)) for doc_id, score in scored[:top_k]]


# ============================================================
# Compare: no reranking vs Cohere vs local cross-encoder
# ============================================================

TEST_QUERIES = [
    "What should I do when a service is running slowly?",
    "How do I restart the matching engine after a config change?",
    "What are the steps for incident escalation?",
    "KAFKA_BROKER_DOWN",
]


def compare_rerankers(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    top_k: int = 5,
    candidates: int = 20,
):
    """Run retrieval with and without reranking, show the difference."""
    for query in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        # Stage 1: hybrid retrieval
        candidate_ids = hybrid_search(collection, bm25_index, query, top_k=candidates)

        # No reranking — just take top-K from hybrid
        print(f"\n  [HYBRID ONLY — no reranking]")
        for i, doc_id in enumerate(candidate_ids[:top_k], 1):
            text = get_chunk_text(chunks, doc_id)[:80]
            print(f"    {i}. {text}...")

        # Cohere reranking
        try:
            t0 = time.time()
            cohere_results = cohere_rerank(query, candidate_ids, chunks, top_k=top_k)
            cohere_ms = (time.time() - t0) * 1000
            print(f"\n  [COHERE RERANK] ({cohere_ms:.0f}ms)")
            for i, (doc_id, score) in enumerate(cohere_results, 1):
                text = get_chunk_text(chunks, doc_id)[:80]
                print(f"    {i}. (score={score:.4f}) {text}...")
        except Exception as e:
            print(f"\n  [COHERE RERANK] Skipped — {e}")

        # Local cross-encoder reranking
        t0 = time.time()
        local_results = local_rerank(query, candidate_ids, chunks, top_k=top_k)
        local_ms = (time.time() - t0) * 1000
        print(f"\n  [LOCAL CROSS-ENCODER] ({local_ms:.0f}ms)")
        for i, (doc_id, score) in enumerate(local_results, 1):
            text = get_chunk_text(chunks, doc_id)[:80]
            print(f"    {i}. (score={score:.4f}) {text}...")


# ============================================================
# Main
# ============================================================

def main():
    print("Session 7 — Reranking: Cross-Encoders for Better Retrieval\n")

    chunks = load_and_chunk()
    print()
    collection = build_vector_index(chunks)
    bm25_index = BM25Index(chunks)
    print(f"BM25 index: {len(chunks)} chunks indexed")

    compare_rerankers(chunks, collection, bm25_index, top_k=5, candidates=20)

    print(f"\n\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  1. Reranking reorders the top-N candidates from first-stage retrieval.
  2. Cross-encoders (Cohere, local) see query+doc together → better relevance.
  3. Cohere Rerank: fast API call, high quality, costs money per query.
  4. Local cross-encoder: free, slower, good for prototyping / offline eval.
  5. Typical pipeline: retrieve 20-50 → rerank → keep top 3-5 for generation.
  6. Reranking is often the single biggest quality improvement to a RAG system.
""")


if __name__ == "__main__":
    main()
