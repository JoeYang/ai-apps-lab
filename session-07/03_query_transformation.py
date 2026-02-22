"""
Session 7, Task 3: Query Transformation — HyDE & Multi-Query
==============================================================
Transform user queries before retrieval to improve search quality.

Three techniques:
  1. HyDE — generate a hypothetical answer, embed that instead
  2. Multi-Query — generate multiple reformulations, merge results
  3. Step-Back — generate a broader question for foundational context

Run: python 03_query_transformation.py

Requires: pip install chromadb openai rank-bm25 anthropic
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass

import chromadb
import openai
import anthropic

oai = openai.OpenAI()
claude = anthropic.Anthropic()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5-20250929"
DOCS_DIR = Path(__file__).parent.parent / "session-06" / "docs"


# ============================================================
# Reuse chunking and indexing setup
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
        name="query_transform_demo", metadata={"hnsw:space": "cosine"}
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


def embed_query(query: str) -> list[float]:
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    return resp.data[0].embedding


def vector_search(collection: chromadb.Collection, query: str, top_k: int = 5) -> list[str]:
    results = collection.query(query_embeddings=[embed_query(query)], n_results=top_k)
    return results["ids"][0]


def vector_search_by_embedding(
    collection: chromadb.Collection, embedding: list[float], top_k: int = 5
) -> list[str]:
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["ids"][0]


def get_chunk_text(chunks: list[Chunk], doc_id: str) -> str:
    for c in chunks:
        if c.doc_id == doc_id:
            return c.text
    return "(not found)"


def reciprocal_rank_fusion(*rankings: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_pos, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_pos)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]


# ============================================================
# Technique 1: HyDE — Hypothetical Document Embeddings
# ============================================================

def generate_hypothetical_answer(query: str) -> str:
    """
    Ask an LLM to write a hypothetical answer to the query.
    We embed THIS instead of the raw query for better retrieval.
    """
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"Write a short, factual paragraph that would answer this question. "
                f"Write it as if it's from a technical document. Do not say "
                f"\"I don't know\" — just write a plausible answer.\n\n"
                f"Question: {query}"
            ),
        }],
    )
    answer = response.content[0].text
    return answer


def hyde_search(
    collection: chromadb.Collection, query: str, top_k: int = 5
) -> tuple[list[str], str]:
    """
    HyDE retrieval:
    1. Generate a hypothetical answer
    2. Embed the hypothetical answer
    3. Search with that embedding
    """
    hypothetical = generate_hypothetical_answer(query)
    embedding = embed_query(hypothetical)
    results = vector_search_by_embedding(collection, embedding, top_k=top_k)
    return results, hypothetical


# ============================================================
# Technique 2: Multi-Query — generate reformulations
# ============================================================

def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """
    Ask an LLM to generate N different reformulations of the query.
    Each variant might surface different relevant documents.
    """
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} different search queries that would help answer "
                f"this question. Each should approach the topic from a different "
                f"angle or use different terminology. Return ONLY a JSON array "
                f"of strings, nothing else.\n\n"
                f"Question: {query}"
            ),
        }],
    )
    text = response.content[0].text.strip()
    # Parse JSON array from response
    try:
        variants = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON array if wrapped in markdown code block
        match = re.search(r'\[.*\]', text, re.DOTALL)
        variants = json.loads(match.group()) if match else [query]
    return variants


def multi_query_search(
    collection: chromadb.Collection, query: str, top_k: int = 5, n_variants: int = 3
) -> tuple[list[str], list[str]]:
    """
    Multi-query retrieval:
    1. Generate N query variants
    2. Search with each variant
    3. Fuse results with RRF
    """
    variants = generate_query_variants(query, n=n_variants)

    # Search with original + all variants
    all_queries = [query] + variants
    rankings = []
    for q in all_queries:
        results = vector_search(collection, q, top_k=top_k * 2)
        rankings.append(results)

    fused = reciprocal_rank_fusion(*rankings)
    return fused[:top_k], variants


# ============================================================
# Technique 3: Step-Back Prompting
# ============================================================

def generate_step_back_query(query: str) -> str:
    """
    Generate a more general/abstract version of the query.
    Retrieves foundational context to help answer the specific question.
    """
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                f"Given this specific question, generate a single broader, "
                f"more general question that would retrieve foundational "
                f"knowledge needed to answer it. Return ONLY the question.\n\n"
                f"Specific question: {query}"
            ),
        }],
    )
    return response.content[0].text.strip()


def step_back_search(
    collection: chromadb.Collection, query: str, top_k: int = 5
) -> tuple[list[str], str]:
    """
    Step-back retrieval:
    1. Generate a broader question
    2. Search with both original and step-back queries
    3. Fuse results
    """
    step_back = generate_step_back_query(query)

    original_results = vector_search(collection, query, top_k=top_k * 2)
    stepback_results = vector_search(collection, step_back, top_k=top_k * 2)

    fused = reciprocal_rank_fusion(original_results, stepback_results)
    return fused[:top_k], step_back


# ============================================================
# Compare all approaches
# ============================================================

TEST_QUERIES = [
    # Vague query — HyDE and multi-query should help
    "deployment issues",
    # Specific but uses different vocabulary than docs might
    "service keeps crashing under load",
    # Short query that benefits from expansion
    "config changes",
]


def compare_approaches(chunks: list[Chunk], collection: chromadb.Collection):
    for query in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY: \"{query}\"")
        print(f"{'='*70}")

        # Baseline: plain vector search
        baseline = vector_search(collection, query, top_k=3)
        print(f"\n  [BASELINE — plain vector search]")
        for i, doc_id in enumerate(baseline, 1):
            text = get_chunk_text(chunks, doc_id)[:80]
            print(f"    {i}. {text}...")

        # HyDE
        hyde_results, hypothetical = hyde_search(collection, query, top_k=3)
        print(f"\n  [HyDE — hypothetical document]")
        print(f"    Generated: \"{hypothetical[:100]}...\"")
        for i, doc_id in enumerate(hyde_results, 1):
            text = get_chunk_text(chunks, doc_id)[:80]
            print(f"    {i}. {text}...")

        # Multi-Query
        mq_results, variants = multi_query_search(collection, query, top_k=3)
        print(f"\n  [MULTI-QUERY — {len(variants)} variants]")
        for v in variants:
            print(f"    Variant: \"{v}\"")
        for i, doc_id in enumerate(mq_results, 1):
            text = get_chunk_text(chunks, doc_id)[:80]
            print(f"    {i}. {text}...")

        # Step-Back
        sb_results, step_back = step_back_search(collection, query, top_k=3)
        print(f"\n  [STEP-BACK]")
        print(f"    Step-back query: \"{step_back}\"")
        for i, doc_id in enumerate(sb_results, 1):
            text = get_chunk_text(chunks, doc_id)[:80]
            print(f"    {i}. {text}...")


# ============================================================
# Main
# ============================================================

def main():
    print("Session 7 — Query Transformation: HyDE, Multi-Query, Step-Back\n")

    chunks = load_and_chunk()
    print()
    collection = build_vector_index(chunks)

    compare_approaches(chunks, collection)

    print(f"\n\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  1. HyDE: embed a hypothetical answer instead of the raw query.
     Best for vague queries. Can hurt on exact-match lookups.

  2. Multi-Query: generate N reformulations, search all, fuse with RRF.
     Casts a wider net. Costs N extra LLM calls + N extra searches.

  3. Step-Back: generate a broader question for foundational context.
     Good for specific questions that need background knowledge.

  4. All three add LLM calls (latency + cost) before retrieval.
     Use them when baseline retrieval quality isn't good enough.

  5. These compose with hybrid search and reranking:
     query transform → hybrid retrieve → rerank → generate
""")


if __name__ == "__main__":
    main()
