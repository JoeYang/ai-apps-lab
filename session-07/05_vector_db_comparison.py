"""
Session 7, Task 5: Vector DB Comparison — Chroma vs Qdrant
============================================================
Set up Qdrant as an alternative to Chroma and compare:
  - Indexing speed
  - Query latency
  - Feature differences (filtering, hybrid search)

We skip pgvector here (requires a running Postgres instance) but
discuss it in the session notes.

Run: python 05_vector_db_comparison.py

Requires:
  pip install chromadb openai qdrant-client

Note: Qdrant runs in-memory here (no Docker needed for this demo).
For production, run `docker run -p 6333:6333 qdrant/qdrant`.
"""

import re
import time
from pathlib import Path
from dataclasses import dataclass

import chromadb
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

oai = openai.OpenAI()

EMBED_MODEL = "text-embedding-3-small"
DOCS_DIR = Path(__file__).parent.parent / "session-06" / "docs"


# ============================================================
# Shared: loading and chunking
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts with OpenAI."""
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = oai.embeddings.create(input=batch, model=EMBED_MODEL)
        all_embeddings.extend([e.embedding for e in resp.data])
    return all_embeddings


def embed_query(query: str) -> list[float]:
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    return resp.data[0].embedding


# ============================================================
# Chroma setup
# ============================================================

def setup_chroma(chunks: list[Chunk], embeddings: list[list[float]]) -> chromadb.Collection:
    client = chromadb.Client()
    collection = client.create_collection(
        name="comparison_chroma", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c.doc_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source} for c in chunks],
    )
    return collection


def search_chroma(collection: chromadb.Collection, query_emb: list[float], top_k: int = 5) -> list[dict]:
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    return [
        {"id": id_, "text": doc, "source": meta["source"]}
        for id_, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]


def search_chroma_filtered(
    collection: chromadb.Collection, query_emb: list[float], source: str, top_k: int = 5
) -> list[dict]:
    """Chroma metadata filtering."""
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where={"source": source},
    )
    return [
        {"id": id_, "text": doc, "source": meta["source"]}
        for id_, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]


# ============================================================
# Qdrant setup
# ============================================================

def setup_qdrant(chunks: list[Chunk], embeddings: list[list[float]]) -> QdrantClient:
    # In-memory mode — no Docker required for demo
    client = QdrantClient(":memory:")

    # Get embedding dimension from first embedding
    dim = len(embeddings[0])

    client.create_collection(
        collection_name="comparison_qdrant",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Upsert points
    points = [
        PointStruct(
            id=i,
            vector=emb,
            payload={
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "source": chunk.source,
            },
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    # Qdrant supports batch upsert
    client.upsert(collection_name="comparison_qdrant", points=points)
    return client


def search_qdrant(client: QdrantClient, query_emb: list[float], top_k: int = 5) -> list[dict]:
    results = client.query_points(
        collection_name="comparison_qdrant",
        query=query_emb,
        limit=top_k,
    )
    return [
        {
            "id": hit.payload["doc_id"],
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "score": hit.score,
        }
        for hit in results.points
    ]


def search_qdrant_filtered(
    client: QdrantClient, query_emb: list[float], source: str, top_k: int = 5
) -> list[dict]:
    """Qdrant payload filtering — richer than Chroma's metadata filtering."""
    results = client.query_points(
        collection_name="comparison_qdrant",
        query=query_emb,
        query_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
        limit=top_k,
    )
    return [
        {
            "id": hit.payload["doc_id"],
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "score": hit.score,
        }
        for hit in results.points
    ]


# ============================================================
# Benchmarks
# ============================================================

TEST_QUERIES = [
    "What should I do when a service is running slowly?",
    "How do I restart the matching engine after a config change?",
    "KAFKA_BROKER_DOWN",
    "What are the steps for incident escalation?",
]


def benchmark_indexing(chunks: list[Chunk], embeddings: list[list[float]]):
    """Compare indexing speed."""
    print(f"\n{'='*70}")
    print("BENCHMARK: Indexing Speed")
    print(f"{'='*70}")

    t0 = time.time()
    chroma_col = setup_chroma(chunks, embeddings)
    chroma_ms = (time.time() - t0) * 1000
    print(f"  Chroma:  {chroma_ms:.1f}ms to index {len(chunks)} chunks")

    t0 = time.time()
    qdrant_client = setup_qdrant(chunks, embeddings)
    qdrant_ms = (time.time() - t0) * 1000
    print(f"  Qdrant:  {qdrant_ms:.1f}ms to index {len(chunks)} chunks")

    return chroma_col, qdrant_client


def benchmark_search(chroma_col: chromadb.Collection, qdrant_client: QdrantClient):
    """Compare query latency and results."""
    print(f"\n{'='*70}")
    print("BENCHMARK: Search Latency & Results")
    print(f"{'='*70}")

    for query in TEST_QUERIES:
        query_emb = embed_query(query)
        print(f"\n  Query: \"{query}\"")

        # Chroma
        t0 = time.time()
        chroma_results = search_chroma(chroma_col, query_emb, top_k=3)
        chroma_us = (time.time() - t0) * 1_000_000
        print(f"\n    [Chroma] ({chroma_us:.0f}us)")
        for i, r in enumerate(chroma_results, 1):
            print(f"      {i}. [{r['source']}] {r['text'][:60]}...")

        # Qdrant
        t0 = time.time()
        qdrant_results = search_qdrant(qdrant_client, query_emb, top_k=3)
        qdrant_us = (time.time() - t0) * 1_000_000
        print(f"\n    [Qdrant] ({qdrant_us:.0f}us)")
        for i, r in enumerate(qdrant_results, 1):
            print(f"      {i}. (score={r['score']:.4f}) [{r['source']}] {r['text'][:60]}...")


def benchmark_filtering(chroma_col: chromadb.Collection, qdrant_client: QdrantClient):
    """Compare metadata/payload filtering."""
    print(f"\n{'='*70}")
    print("BENCHMARK: Filtered Search")
    print(f"{'='*70}")

    query = "How do I handle an incident?"
    query_emb = embed_query(query)
    filter_source = "runbooks.md"

    print(f"\n  Query: \"{query}\"")
    print(f"  Filter: source = \"{filter_source}\"")

    # Chroma filtered
    chroma_results = search_chroma_filtered(chroma_col, query_emb, filter_source, top_k=3)
    print(f"\n    [Chroma filtered]")
    for i, r in enumerate(chroma_results, 1):
        print(f"      {i}. [{r['source']}] {r['text'][:60]}...")

    # Qdrant filtered
    qdrant_results = search_qdrant_filtered(qdrant_client, query_emb, filter_source, top_k=3)
    print(f"\n    [Qdrant filtered]")
    for i, r in enumerate(qdrant_results, 1):
        print(f"      {i}. (score={r['score']:.4f}) [{r['source']}] {r['text'][:60]}...")

    print(f"\n  Note: Qdrant supports richer filters (range, geo, nested) via its")
    print(f"  Filter API. Chroma uses simple dict-based where clauses.")


# ============================================================
# Main
# ============================================================

def main():
    print("Session 7 — Vector DB Comparison: Chroma vs Qdrant\n")

    # Load, chunk, and embed once (shared across both DBs)
    chunks = load_and_chunk()
    print("\nEmbedding chunks (shared across both databases)...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    print(f"Embedded {len(embeddings)} chunks ({len(embeddings[0])} dimensions)\n")

    # Run benchmarks
    chroma_col, qdrant_client = benchmark_indexing(chunks, embeddings)
    benchmark_search(chroma_col, qdrant_client)
    benchmark_filtering(chroma_col, qdrant_client)

    print(f"\n\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print("""
  Chroma:
    + Zero setup (pip install, in-memory or local file)
    + Great for prototyping and learning
    - No native hybrid search
    - Limited filtering
    - Single-node only

  Qdrant:
    + Purpose-built for vector search (Rust, fast)
    + Native hybrid search (sparse + dense vectors)
    + Rich payload filtering (range, geo, nested, match)
    + Production features: snapshots, replication, multi-tenancy
    - Extra service to manage (Docker or cloud)

  pgvector (not demoed — needs running Postgres):
    + Lives inside your existing Postgres
    + Full SQL power: JOINs, WHERE, transactions, ACID
    + Familiar tooling (pg_dump, replication, monitoring)
    - Slower at pure vector search than Qdrant
    - HNSW/IVFFlat index tuning is manual
    - Vector operations compete with your other queries

  Rule of thumb:
    Prototyping → Chroma
    Already have Postgres → pgvector
    Production vector search → Qdrant (or Pinecone, Weaviate)
""")


if __name__ == "__main__":
    main()
