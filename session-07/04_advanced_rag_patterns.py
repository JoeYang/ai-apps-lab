"""
Session 7, Task 4: Advanced RAG Patterns
==========================================
Implement three production-grade RAG patterns:

  1. Parent-Document Retrieval — index small, retrieve big
  2. Corrective RAG (CRAG) — grade retrieved docs, fallback if irrelevant
  3. Adaptive RAG — route queries to different strategies

Run: python 04_advanced_rag_patterns.py

Requires: pip install chromadb openai anthropic
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
# Data structures
# ============================================================

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    doc_id: str = ""
    parent_id: str = ""  # links child → parent chunk

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = f"{self.source}::chunk-{self.chunk_id}"


# ============================================================
# Pattern 1: Parent-Document Retrieval
# ============================================================

def load_with_parent_chunks(
    max_child_chars: int = 200, max_parent_chars: int = 800
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Create two sets of chunks from the same documents:
      - child chunks: small (200 chars) — good for precise embedding matches
      - parent chunks: large (800 chars) — good for LLM context

    Each child stores its parent_id so we can look up the parent at
    retrieval time.
    """
    parents = []
    children = []

    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text()
        source = path.name

        # Create parent chunks (large)
        parent_chunks = _split_text(text, source, max_parent_chars, prefix="parent")
        parents.extend(parent_chunks)

        # Create child chunks (small) within each parent
        for parent in parent_chunks:
            child_chunks = _split_text(
                parent.text, source, max_child_chars, prefix="child"
            )
            for child in child_chunks:
                child.parent_id = parent.doc_id
            children.extend(child_chunks)

    print(f"Parent-Doc: {len(parents)} parents, {len(children)} children")
    return parents, children


def _split_text(text: str, source: str, max_chars: int, prefix: str) -> list[Chunk]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, idx = [], "", 0
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(Chunk(
                text=current.strip(), source=source, chunk_id=idx,
                doc_id=f"{source}::{prefix}-{idx}",
            ))
            idx += 1
            current = ""
        current += sent + " "
    if current.strip():
        chunks.append(Chunk(
            text=current.strip(), source=source, chunk_id=idx,
            doc_id=f"{source}::{prefix}-{idx}",
        ))
    return chunks


def build_child_index(children: list[Chunk]) -> chromadb.Collection:
    """Index only the small child chunks for retrieval."""
    client = chromadb.Client()
    collection = client.create_collection(
        name="parent_doc_demo", metadata={"hnsw:space": "cosine"}
    )
    texts = [c.text for c in children]
    ids = [c.doc_id for c in children]
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = oai.embeddings.create(input=batch, model=EMBED_MODEL)
        embeddings.extend([e.embedding for e in resp.data])
    collection.add(
        ids=ids, documents=texts, embeddings=embeddings,
        metadatas=[{"source": c.source, "parent_id": c.parent_id} for c in children],
    )
    return collection


def parent_document_search(
    collection: chromadb.Collection,
    parents: list[Chunk],
    query: str,
    top_k: int = 3,
) -> list[Chunk]:
    """
    Search child chunks, but return their parent chunks.
    This gives precise matching + rich context.
    """
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    query_emb = resp.data[0].embedding

    # Retrieve more children than needed since multiple children may share a parent
    results = collection.query(
        query_embeddings=[query_emb], n_results=top_k * 3
    )

    # Map child hits → unique parents
    parent_lookup = {p.doc_id: p for p in parents}
    seen_parents = set()
    parent_results = []

    for metadata in results["metadatas"][0]:
        pid = metadata["parent_id"]
        if pid not in seen_parents and pid in parent_lookup:
            seen_parents.add(pid)
            parent_results.append(parent_lookup[pid])
            if len(parent_results) >= top_k:
                break

    return parent_results


def demo_parent_document():
    print(f"\n{'='*70}")
    print("PATTERN 1: Parent-Document Retrieval")
    print(f"{'='*70}")

    parents, children = load_with_parent_chunks()
    collection = build_child_index(children)

    query = "How do I handle a service outage?"
    print(f"\n  Query: \"{query}\"")

    # Show what child chunks match
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    child_results = collection.query(query_embeddings=[resp.data[0].embedding], n_results=3)
    print(f"\n  [CHILD matches — small, precise]")
    for i, doc in enumerate(child_results["documents"][0], 1):
        print(f"    {i}. ({len(doc)} chars) {doc[:80]}...")

    # Show the parent chunks we actually return
    parent_results = parent_document_search(collection, parents, query, top_k=3)
    print(f"\n  [PARENT chunks returned — large, contextual]")
    for i, p in enumerate(parent_results, 1):
        print(f"    {i}. ({len(p.text)} chars) {p.text[:80]}...")

    print(f"\n  → Child chunks give precise matches")
    print(f"  → Parent chunks give the LLM enough context to answer well")


# ============================================================
# Pattern 2: Corrective RAG (CRAG)
# ============================================================

def grade_document(query: str, document: str) -> dict:
    """
    Use an LLM to grade whether a retrieved document is relevant.
    Returns {"relevant": bool, "reason": str}
    """
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                f"You are a relevance grader. Given a user question and a "
                f"retrieved document, determine if the document contains "
                f"information relevant to answering the question.\n\n"
                f"Question: {query}\n\n"
                f"Document: {document}\n\n"
                f"Return ONLY a JSON object: "
                f'{{\"relevant\": true/false, \"reason\": \"brief explanation\"}}'
            ),
        }],
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group()) if match else {"relevant": False, "reason": "parse error"}


def corrective_rag(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    query: str,
    top_k: int = 5,
) -> dict:
    """
    Corrective RAG pipeline:
    1. Retrieve top-K documents
    2. Grade each for relevance
    3. Keep only relevant ones
    4. If none are relevant, flag for fallback
    """
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    results = collection.query(query_embeddings=[resp.data[0].embedding], n_results=top_k)

    graded = []
    for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
        grade = grade_document(query, doc_text)
        graded.append({
            "doc_id": doc_id,
            "text": doc_text,
            "relevant": grade.get("relevant", False),
            "reason": grade.get("reason", ""),
        })

    relevant = [g for g in graded if g["relevant"]]
    irrelevant = [g for g in graded if not g["relevant"]]

    if not relevant:
        action = "FALLBACK — no relevant docs found, try web search or rephrase"
    elif len(relevant) < len(graded) / 2:
        action = "PARTIAL — some relevant docs, generate with filtered set"
    else:
        action = "PROCEED — most docs are relevant, generate normally"

    return {
        "action": action,
        "relevant": relevant,
        "irrelevant": irrelevant,
        "all_graded": graded,
    }


def demo_corrective_rag():
    print(f"\n{'='*70}")
    print("PATTERN 2: Corrective RAG (CRAG)")
    print(f"{'='*70}")

    # Build a simple index for this demo
    chunks = _load_simple_chunks()
    collection = _build_simple_index(chunks, "crag_demo")

    queries = [
        # Should find relevant docs
        "What are the incident escalation steps?",
        # Might find tangentially related but not truly relevant docs
        "What is the company's vacation policy?",
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        result = corrective_rag(collection, chunks, query, top_k=3)
        print(f"  Action: {result['action']}")
        print(f"  Relevant: {len(result['relevant'])}, Irrelevant: {len(result['irrelevant'])}")
        for g in result["all_graded"]:
            status = "RELEVANT" if g["relevant"] else "IRRELEVANT"
            print(f"    [{status}] {g['reason']}")
            print(f"             {g['text'][:60]}...")


# ============================================================
# Pattern 3: Adaptive RAG — route queries to strategies
# ============================================================

def classify_query(query: str) -> dict:
    """
    Classify a query to determine the best retrieval strategy.
    """
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                f"Classify this query into one of these categories:\n"
                f"- \"direct\": Can be answered directly by the LLM without any documents "
                f"(general knowledge, math, definitions)\n"
                f"- \"simple_rag\": Needs document retrieval, single straightforward lookup\n"
                f"- \"multi_step\": Complex question requiring multiple retrievals or "
                f"reasoning across documents\n"
                f"- \"clarification\": Too ambiguous to answer, needs user clarification\n\n"
                f"Query: {query}\n\n"
                f"Return ONLY a JSON object: "
                f'{{\"category\": \"...\", \"reason\": \"brief explanation\", '
                f'\"sub_queries\": [\"...\"] }}\n'
                f"Include sub_queries only for multi_step category."
            ),
        }],
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group()) if match else {"category": "simple_rag", "reason": "parse error"}


def adaptive_rag(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    query: str,
) -> dict:
    """
    Route the query to the appropriate strategy based on classification.
    """
    classification = classify_query(query)
    category = classification.get("category", "simple_rag")

    if category == "direct":
        # No retrieval needed — answer directly
        response = claude.messages.create(
            model=CHAT_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": query}],
        )
        return {
            "strategy": "direct (no retrieval)",
            "classification": classification,
            "answer": response.content[0].text,
        }

    elif category == "clarification":
        return {
            "strategy": "clarification needed",
            "classification": classification,
            "answer": f"I need more context: {classification.get('reason', '')}",
        }

    elif category == "multi_step":
        # Decompose and retrieve for each sub-query
        sub_queries = classification.get("sub_queries", [query])
        all_context = []
        resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)

        for sq in sub_queries:
            sq_resp = oai.embeddings.create(input=[sq], model=EMBED_MODEL)
            results = collection.query(
                query_embeddings=[sq_resp.data[0].embedding], n_results=2
            )
            all_context.extend(results["documents"][0])

        context = "\n\n".join(all_context)
        response = claude.messages.create(
            model=CHAT_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }],
        )
        return {
            "strategy": f"multi_step ({len(sub_queries)} sub-queries)",
            "classification": classification,
            "answer": response.content[0].text,
        }

    else:  # simple_rag
        resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
        results = collection.query(
            query_embeddings=[resp.data[0].embedding], n_results=3
        )
        context = "\n\n".join(results["documents"][0])
        response = claude.messages.create(
            model=CHAT_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }],
        )
        return {
            "strategy": "simple_rag",
            "classification": classification,
            "answer": response.content[0].text,
        }


def demo_adaptive_rag():
    print(f"\n{'='*70}")
    print("PATTERN 3: Adaptive RAG — Route Queries to Strategies")
    print(f"{'='*70}")

    chunks = _load_simple_chunks()
    collection = _build_simple_index(chunks, "adaptive_demo")

    queries = [
        # Direct — no retrieval needed
        "What does HTTP 503 mean?",
        # Simple RAG — single lookup
        "What is the escalation process for incidents?",
        # Multi-step — needs multiple lookups
        "Compare the monitoring setup in the runbook with the system config settings",
        # Ambiguous — needs clarification
        "fix it",
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        result = adaptive_rag(collection, chunks, query)
        print(f"  Strategy: {result['strategy']}")
        print(f"  Reason: {result['classification'].get('reason', 'n/a')}")
        print(f"  Answer: {result['answer'][:120]}...")


# ============================================================
# Helpers
# ============================================================

def _load_simple_chunks() -> list[Chunk]:
    all_chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text()
        chunks = _split_text(text, path.name, 500, "chunk")
        all_chunks.extend(chunks)
    return all_chunks


def _build_simple_index(chunks: list[Chunk], name: str) -> chromadb.Collection:
    client = chromadb.Client()
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
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
# Main
# ============================================================

def main():
    print("Session 7 — Advanced RAG Patterns\n")

    demo_parent_document()
    demo_corrective_rag()
    demo_adaptive_rag()

    print(f"\n\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  1. Parent-Document Retrieval: index small chunks for precise matching,
     return large parent chunks for rich LLM context. Best of both worlds.

  2. Corrective RAG: grade retrieved docs for relevance before generating.
     Prevents hallucination from irrelevant context. Falls back gracefully.

  3. Adaptive RAG: classify the query first, route to the right strategy.
     Saves cost/latency on simple queries, handles complex ones properly.

  4. Self-RAG: model decides when to retrieve and checks its own output.
     Requires a specially trained model (not just prompting).

  5. Agentic RAG: wrap RAG in an agent loop — retrieve, reason, retrieve
     again. Most flexible, highest cost. You'll build this in Phase 3.

  These patterns compose: Adaptive routing → Parent-doc retrieval →
  Corrective grading → Reranking → Generation.
""")


if __name__ == "__main__":
    main()
