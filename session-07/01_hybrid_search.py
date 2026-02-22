"""
Session 7, Task 1: Hybrid Search — Vector + BM25
==================================================
Combine semantic vector search (Chroma + OpenAI embeddings) with
keyword-based BM25 search, then fuse results using Reciprocal Rank
Fusion (RRF).

This demonstrates why hybrid search beats either approach alone —
vectors handle meaning, BM25 handles exact terms.

Run: python 01_hybrid_search.py

Requires: pip install chromadb openai rank-bm25 anthropic
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
import openai
from rank_bm25 import BM25Okapi

oai = openai.OpenAI()

EMBED_MODEL = "text-embedding-3-small"
DOCS_DIR = Path(__file__).parent.parent / "session-06" / "docs"


# ============================================================
# Step 1: LOAD & CHUNK — reuse the simple sentence-aware chunker
# ============================================================

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    doc_id: str = ""  # unique id for retrieval

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = f"{self.source}::chunk-{self.chunk_id}"


def sentence_chunk(text: str, source: str, max_chars: int = 500) -> list[Chunk]:
    """Split text on sentence boundaries, respecting a max size."""
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
    """Load all markdown docs from the docs folder and chunk them."""
    all_chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text()
        chunks = sentence_chunk(text, source=path.name)
        all_chunks.extend(chunks)
    print(f"Loaded {len(all_chunks)} chunks from {len(list(DOCS_DIR.glob('*.md')))} documents")
    return all_chunks


# ============================================================
# Step 2: BUILD INDEXES — one vector (Chroma), one keyword (BM25)
# ============================================================

def build_vector_index(chunks: list[Chunk]) -> chromadb.Collection:
    """Embed chunks and store in Chroma."""
    client = chromadb.Client()  # in-memory
    collection = client.create_collection(
        name="hybrid_demo",
        metadata={"hnsw:space": "cosine"},
    )

    # Batch embed with OpenAI
    texts = [c.text for c in chunks]
    ids = [c.doc_id for c in chunks]

    # OpenAI embedding API allows up to ~8k inputs, batch in groups of 100
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = oai.embeddings.create(input=batch, model=EMBED_MODEL)
        all_embeddings.extend([e.embedding for e in resp.data])

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=all_embeddings,
        metadatas=[{"source": c.source} for c in chunks],
    )
    print(f"Vector index: {collection.count()} chunks indexed in Chroma")
    return collection


class BM25Index:
    """Thin wrapper around rank_bm25 that maps back to chunk IDs."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.doc_ids = [c.doc_id for c in chunks]
        # Tokenize: lowercase, split on non-alphanumeric
        self.tokenized = [re.split(r'\W+', c.text.lower()) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        print(f"BM25 index: {len(chunks)} chunks indexed")

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return list of (doc_id, score) sorted by score descending."""
        tokens = re.split(r'\W+', query.lower())
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]


# ============================================================
# Step 3: SEARCH — vector, BM25, and hybrid with RRF
# ============================================================

def embed_query(query: str) -> list[float]:
    """Embed a single query string using the same OpenAI model as indexing."""
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    return resp.data[0].embedding


def vector_search(
    collection: chromadb.Collection, query: str, top_k: int = 20
) -> list[str]:
    """Return ranked doc IDs from vector search."""
    results = collection.query(query_embeddings=[embed_query(query)], n_results=top_k)
    return results["ids"][0]


def bm25_search(index: BM25Index, query: str, top_k: int = 20) -> list[str]:
    """Return ranked doc IDs from BM25 search."""
    results = index.search(query, top_k=top_k)
    return [doc_id for doc_id, _ in results]


def reciprocal_rank_fusion(
    *rankings: list[str], k: int = 60
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) — combine multiple ranked lists.

    For each document, the RRF score is:
        sum over each ranking of: 1 / (k + rank)

    where rank is 1-based position. Documents appearing in multiple
    lists get boosted. k=60 is the standard default from the original
    RRF paper (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_pos, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_pos)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# Step 4: COMPARE — run all three approaches on test queries
# ============================================================

TEST_QUERIES = [
    # Semantic query — vector search should shine
    "What should I do when a service is running slowly?",
    # Exact term query — BM25 should shine
    "KAFKA_BROKER_DOWN",
    # Mixed query — hybrid should win
    "How do I restart the matching engine after a config change?",
    # Another exact match case
    "max_connections_per_host",
    # Conceptual question
    "What are the steps for incident escalation?",
]


def get_chunk_text(chunks: list[Chunk], doc_id: str) -> str:
    """Look up chunk text by doc_id."""
    for c in chunks:
        if c.doc_id == doc_id:
            return c.text
    return "(not found)"


def compare_searches(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    top_k: int = 5,
):
    """Run vector, BM25, and hybrid search side by side."""
    for query in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        vec_ids = vector_search(collection, query, top_k=20)
        bm25_ids = bm25_search(bm25_index, query, top_k=20)
        hybrid_ranked = reciprocal_rank_fusion(vec_ids, bm25_ids, k=60)

        for label, doc_ids in [
            ("VECTOR", vec_ids[:top_k]),
            ("BM25", bm25_ids[:top_k]),
            ("HYBRID (RRF)", [doc_id for doc_id, _ in hybrid_ranked[:top_k]]),
        ]:
            print(f"\n  [{label}]")
            for i, doc_id in enumerate(doc_ids, 1):
                text_preview = get_chunk_text(chunks, doc_id)[:80]
                print(f"    {i}. [{doc_id}]")
                print(f"       {text_preview}...")


# ============================================================
# Step 5: WEIGHTED HYBRID — alternative fusion method
# ============================================================

def weighted_hybrid_search(
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    query: str,
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Weighted linear combination of normalized vector and BM25 scores.

    alpha = 1.0 → pure vector search
    alpha = 0.0 → pure BM25 search
    alpha = 0.5 → equal weight (good default)
    """
    # Vector scores (Chroma returns distances; convert to similarity)
    vec_results = collection.query(query_embeddings=[embed_query(query)], n_results=top_k * 2)
    vec_ids = vec_results["ids"][0]
    vec_distances = vec_results["distances"][0]
    # Cosine distance → similarity: sim = 1 - distance
    vec_sims = {
        doc_id: 1.0 - dist for doc_id, dist in zip(vec_ids, vec_distances)
    }

    # BM25 scores (already similarity-like, but need normalizing)
    bm25_results = bm25_index.search(query, top_k=top_k * 2)
    max_bm25 = bm25_results[0][1] if bm25_results and bm25_results[0][1] > 0 else 1.0
    bm25_sims = {doc_id: score / max_bm25 for doc_id, score in bm25_results}

    # Combine
    all_doc_ids = set(vec_sims.keys()) | set(bm25_sims.keys())
    combined = {}
    for doc_id in all_doc_ids:
        v_score = vec_sims.get(doc_id, 0.0)
        b_score = bm25_sims.get(doc_id, 0.0)
        combined[doc_id] = alpha * v_score + (1 - alpha) * b_score

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ============================================================
# Main
# ============================================================

def main():
    print("Session 7 — Hybrid Search: Vector + BM25\n")

    # Load and chunk documents
    chunks = load_and_chunk()

    # Build both indexes
    print()
    collection = build_vector_index(chunks)
    bm25_index = BM25Index(chunks)

    # Compare all three search approaches
    compare_searches(chunks, collection, bm25_index, top_k=5)

    # Demo weighted hybrid with different alpha values
    print(f"\n\n{'='*70}")
    print("BONUS: Weighted hybrid search with different alpha values")
    print(f"{'='*70}")

    demo_query = "How do I restart the matching engine after a config change?"
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        results = weighted_hybrid_search(
            collection, bm25_index, demo_query, alpha=alpha, top_k=3
        )
        print(f"\n  alpha={alpha} ({'pure BM25' if alpha == 0 else 'pure vector' if alpha == 1 else 'hybrid'}):")
        for doc_id, score in results:
            print(f"    {score:.4f}  {doc_id}")


if __name__ == "__main__":
    main()
