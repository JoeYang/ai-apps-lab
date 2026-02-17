"""
Session 5: Chroma vs pgvector Benchmark
========================================
Compare vector search performance between:
- Chroma (HNSW, in-memory)
- pgvector with brute force (exact, sequential scan)
- pgvector with HNSW index (approximate)

We generate synthetic embeddings (no API calls) to test at scale:
- 1,000 / 10,000 / 50,000 documents
- Measure: index time, query time, recall accuracy

Run: python 03_chroma_vs_pgvector.py
"""

import random
import time
import math

import chromadb
import psycopg2
from psycopg2.extras import execute_values

# Reproducible results
random.seed(42)

DB_CONFIG = {
    "host": "localhost",
    "user": "postgres",
    "password": "postgres",
    "dbname": "postgres",
}

DIMENSIONS = 1024
QUERY_COUNT = 20


# ============================================================
# Helpers
# ============================================================

def generate_vectors(n: int, dims: int) -> list[list[float]]:
    """Generate n random unit vectors."""
    vectors = []
    for _ in range(n):
        vec = [random.gauss(0, 1) for _ in range(dims)]
        magnitude = math.sqrt(sum(x * x for x in vec))
        vectors.append([x / magnitude for x in vec])
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def brute_force_topk(query: list[float], docs: list[list[float]], k: int) -> list[int]:
    """Ground truth: brute force top-k by cosine similarity."""
    sims = [(i, cosine_similarity(query, d)) for i, d in enumerate(docs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in sims[:k]]


def recall_at_k(predicted: list, actual: list) -> float:
    return len(set(predicted) & set(actual)) / len(actual)


# ============================================================
# Chroma benchmark
# ============================================================

def bench_chroma(doc_vectors: list[list[float]], query_vectors: list[list[float]], k: int):
    client = chromadb.Client()

    # Clean up if exists
    try:
        client.delete_collection("bench")
    except Exception:
        pass

    collection = client.create_collection(
        name="bench",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(i) for i in range(len(doc_vectors))]

    # Index
    start = time.time()
    # Chroma has a batch size limit, add in chunks
    batch_size = 5000
    for i in range(0, len(doc_vectors), batch_size):
        end = min(i + batch_size, len(doc_vectors))
        collection.add(
            ids=ids[i:end],
            embeddings=doc_vectors[i:end],
        )
    index_time = time.time() - start

    # Query
    query_times = []
    all_results = []
    for qv in query_vectors:
        start = time.time()
        results = collection.query(query_embeddings=[qv], n_results=k)
        query_times.append(time.time() - start)
        all_results.append([int(x) for x in results["ids"][0]])

    client.delete_collection("bench")

    return {
        "index_time": index_time,
        "avg_query_ms": sum(query_times) / len(query_times) * 1000,
        "p99_query_ms": sorted(query_times)[int(len(query_times) * 0.99)] * 1000,
        "results": all_results,
    }


# ============================================================
# pgvector benchmark
# ============================================================

def bench_pgvector(doc_vectors: list[list[float]], query_vectors: list[list[float]], k: int, use_hnsw: bool):
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Setup
    cur.execute("DROP TABLE IF EXISTS bench_vectors;")
    cur.execute(f"""
        CREATE TABLE bench_vectors (
            id INTEGER PRIMARY KEY,
            embedding vector({DIMENSIONS})
        );
    """)

    # Index documents
    start = time.time()
    data = [(i, str(v)) for i, v in enumerate(doc_vectors)]
    execute_values(cur, "INSERT INTO bench_vectors (id, embedding) VALUES %s", data)

    if use_hnsw:
        cur.execute("""
            CREATE INDEX ON bench_vectors
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

    index_time = time.time() - start

    # Query
    query_times = []
    all_results = []

    # Set search parameters for HNSW
    if use_hnsw:
        cur.execute("SET hnsw.ef_search = 40;")

    for qv in query_vectors:
        vec_str = str(qv)
        start = time.time()
        cur.execute(f"""
            SELECT id FROM bench_vectors
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vec_str, k))
        rows = cur.fetchall()
        query_times.append(time.time() - start)
        all_results.append([row[0] for row in rows])

    cur.execute("DROP TABLE bench_vectors;")
    cur.close()
    conn.close()

    return {
        "index_time": index_time,
        "avg_query_ms": sum(query_times) / len(query_times) * 1000,
        "p99_query_ms": sorted(query_times)[int(len(query_times) * 0.99)] * 1000,
        "results": all_results,
    }


# ============================================================
# Main benchmark
# ============================================================

if __name__ == "__main__":
    print("=" * 75)
    print("BENCHMARK: Chroma (HNSW) vs pgvector (exact) vs pgvector (HNSW)")
    print(f"Dimensions: {DIMENSIONS}, Queries: {QUERY_COUNT}, Top-K: 10")
    print("=" * 75)

    sizes = [1_000, 10_000, 50_000]

    for n in sizes:
        print(f"\n{'─' * 75}")
        print(f"  DOCUMENTS: {n:,}")
        print(f"{'─' * 75}")

        # Generate data
        print(f"  Generating {n:,} vectors ({DIMENSIONS}d)...", end=" ", flush=True)
        doc_vectors = generate_vectors(n, DIMENSIONS)
        query_vectors = generate_vectors(QUERY_COUNT, DIMENSIONS)
        print("done")

        # Ground truth (brute force in Python)
        print("  Computing ground truth (brute force)...", end=" ", flush=True)
        ground_truth = [brute_force_topk(qv, doc_vectors, 10) for qv in query_vectors]
        print("done")

        # Benchmark each engine
        engines = [
            ("Chroma (HNSW)", lambda dv, qv: bench_chroma(dv, qv, 10)),
            ("pgvector (exact)", lambda dv, qv: bench_pgvector(dv, qv, 10, use_hnsw=False)),
            ("pgvector (HNSW)", lambda dv, qv: bench_pgvector(dv, qv, 10, use_hnsw=True)),
        ]

        results = {}
        for name, bench_fn in engines:
            print(f"  Running {name}...", end=" ", flush=True)
            result = bench_fn(doc_vectors, query_vectors)

            # Calculate recall vs ground truth
            recalls = [
                recall_at_k(result["results"][i], ground_truth[i])
                for i in range(QUERY_COUNT)
            ]
            result["avg_recall"] = sum(recalls) / len(recalls)
            results[name] = result
            print("done")

        # Print results table
        print()
        print(f"  {'Engine':<22} {'Index (s)':<12} {'Avg Query':<14} {'p99 Query':<14} {'Recall@10'}")
        print(f"  {'─'*22} {'─'*12} {'─'*14} {'─'*14} {'─'*10}")
        for name, r in results.items():
            print(
                f"  {name:<22} {r['index_time']:>8.2f}s   "
                f"{r['avg_query_ms']:>9.2f}ms   "
                f"{r['p99_query_ms']:>9.2f}ms   "
                f"{r['avg_recall']:>8.1%}"
            )

    print()
    print("=" * 75)
    print("KEY TAKEAWAYS:")
    print("  1. pgvector (exact) = 100% recall but query time grows linearly with docs")
    print("  2. Chroma (HNSW) = near-perfect recall, query time grows ~O(log n)")
    print("  3. pgvector (HNSW) = same algorithm as Chroma, similar performance")
    print("  4. Index time: HNSW is slower to build (building the graph)")
    print("  5. Choose pgvector if you already use PostgreSQL (one less dependency)")
    print("  6. Choose Chroma for prototyping (zero setup, in-memory)")
    print("  7. At <10K docs, the difference barely matters — pick what's convenient")
    print("=" * 75)
