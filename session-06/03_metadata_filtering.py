"""
Session 6, Task 3: Metadata Filtering & Retrieval Quality
==========================================================
Compare retrieval WITH and WITHOUT metadata filters.

Demonstrates:
1. Unfiltered retrieval (may pull irrelevant doc types)
2. Filtered retrieval (constrain to specific doc types)
3. Quality comparison — does filtering improve or hurt answers?

Run: python 03_metadata_filtering.py
"""

import re
from pathlib import Path

import chromadb
import openai
import anthropic

oai = openai.OpenAI()
claude = anthropic.Anthropic()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5-20250929"
DOCS_DIR = Path(__file__).parent / "docs"


# ============================================================
# Reuse pipeline components from Task 2
# ============================================================

def load_and_chunk() -> list[dict]:
    """Load docs and chunk semantically. Returns list of chunk dicts."""
    chunks = []
    chunk_counter = 0

    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text()
        doc_type = path.stem.replace("-", "_")
        sections = re.split(r'\n(?=#{2,3}\s)', text)

        for section in sections:
            section = section.strip()
            if not section:
                continue
            heading_match = re.match(r'^(#{2,3})\s+(.+?)$', section, re.MULTILINE)
            heading = heading_match.group(2) if heading_match else "Introduction"

            if len(section) // 4 <= 300:
                chunks.append({
                    "id": f"chunk_{chunk_counter:03d}",
                    "text": section,
                    "source": path.name,
                    "heading": heading,
                    "doc_type": doc_type,
                })
                chunk_counter += 1
            else:
                paragraphs = section.split("\n\n")
                current = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    if (len(current) + len(para)) // 4 > 300 and current:
                        chunks.append({
                            "id": f"chunk_{chunk_counter:03d}",
                            "text": current.strip(),
                            "source": path.name,
                            "heading": heading,
                            "doc_type": doc_type,
                        })
                        chunk_counter += 1
                        current = ""
                    current += para + "\n\n"
                if current.strip():
                    chunks.append({
                        "id": f"chunk_{chunk_counter:03d}",
                        "text": current.strip(),
                        "source": path.name,
                        "heading": heading,
                        "doc_type": doc_type,
                    })
                    chunk_counter += 1

    return chunks


def build_index(chunks: list[dict]) -> chromadb.Collection:
    client = chromadb.Client()
    try:
        client.delete_collection("filter_test")
    except Exception:
        pass

    collection = client.create_collection("filter_test", metadata={"hnsw:space": "cosine"})

    embeddings = oai.embeddings.create(
        model=EMBED_MODEL, input=[c["text"] for c in chunks]
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=[e.embedding for e in embeddings.data],
        metadatas=[{
            "source": c["source"],
            "heading": c["heading"],
            "doc_type": c["doc_type"],
        } for c in chunks],
    )

    return collection


def retrieve(collection, query, top_k=5, where=None):
    q_emb = oai.embeddings.create(model=EMBED_MODEL, input=[query])
    results = collection.query(
        query_embeddings=[q_emb.data[0].embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": 1 - results["distances"][0][i],
        })
    return retrieved


RAG_SYSTEM = """You are a trading operations assistant. Answer based ONLY on the provided documents.
If the answer isn't in the documents, say so. Cite sources. Be concise."""


def generate(query, chunks):
    context = "\n\n".join(
        f'<doc id="{c["id"]}" type="{c["metadata"]["doc_type"]}" '
        f'section="{c["metadata"]["heading"]}" sim="{c["similarity"]:.3f}">\n'
        f'{c["text"]}\n</doc>'
        for c in chunks
    )
    resp = claude.messages.create(
        model=CHAT_MODEL, max_tokens=300, system=RAG_SYSTEM,
        messages=[{"role": "user", "content": f"<documents>\n{context}\n</documents>\n\nQuestion: {query}"}],
    )
    return resp.content[0].text


# ============================================================
# Comparison tests
# ============================================================

TEST_CASES = [
    {
        "query": "How do I troubleshoot a FIX session disconnect?",
        "description": "User wants a RUNBOOK, not an incident report",
        "good_filter": {"doc_type": "runbooks"},
        "bad_filter": {"doc_type": "incident_reports"},
    },
    {
        "query": "What caused the order rejection spike in December?",
        "description": "User wants an INCIDENT, not a runbook",
        "good_filter": {"doc_type": "incident_reports"},
        "bad_filter": {"doc_type": "runbooks"},
    },
    {
        "query": "What is the kill switch threshold?",
        "description": "User wants CONFIG data, not incidents",
        "good_filter": {"doc_type": "system_config"},
        "bad_filter": {"doc_type": "incident_reports"},
    },
    {
        "query": "HEDGE_FUND_A risk limits",
        "description": "Specific client info — scattered across docs",
        "good_filter": None,  # Unfiltered is better here — info is in multiple doc types
        "bad_filter": {"doc_type": "runbooks"},
    },
    {
        "query": "How should I handle a fat finger alert?",
        "description": "User wants operational STEPS",
        "good_filter": {"doc_type": "runbooks"},
        "bad_filter": {"doc_type": "system_config"},
    },
]


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 6: Metadata Filtering & Retrieval Quality")
    print("=" * 70)

    # Build index
    print("\nBuilding index...", end=" ", flush=True)
    chunks = load_and_chunk()
    collection = build_index(chunks)
    print(f"done ({collection.count()} chunks)")

    # Show what doc types we have
    doc_types = set(c["doc_type"] for c in chunks)
    type_counts = {t: sum(1 for c in chunks if c["doc_type"] == t) for t in doc_types}
    print(f"Doc types: {type_counts}")

    for test in TEST_CASES:
        print(f"\n{'━' * 70}")
        print(f"  Q: {test['query']}")
        print(f"  Intent: {test['description']}")
        print(f"{'━' * 70}")

        # --- Run 1: No filter ---
        print(f"\n  ▸ NO FILTER (all doc types)")
        unfiltered = retrieve(collection, test["query"], top_k=3)
        for c in unfiltered:
            print(f"    [{c['metadata']['doc_type']:<20}] {c['metadata']['heading'][:40]:<40} sim={c['similarity']:.3f}")
        answer_unfiltered = generate(test["query"], unfiltered)

        # --- Run 2: Good filter ---
        filter_label = test["good_filter"]["doc_type"] if test["good_filter"] else "None (unfiltered is best)"
        print(f"\n  ▸ GOOD FILTER: {filter_label}")
        if test["good_filter"]:
            good_filtered = retrieve(collection, test["query"], top_k=3, where=test["good_filter"])
        else:
            good_filtered = unfiltered  # For cases where unfiltered is best
        for c in good_filtered:
            print(f"    [{c['metadata']['doc_type']:<20}] {c['metadata']['heading'][:40]:<40} sim={c['similarity']:.3f}")
        answer_good = generate(test["query"], good_filtered)

        # --- Run 3: Wrong filter ---
        print(f"\n  ▸ WRONG FILTER: {test['bad_filter']['doc_type']}")
        bad_filtered = retrieve(collection, test["query"], top_k=3, where=test["bad_filter"])
        for c in bad_filtered:
            print(f"    [{c['metadata']['doc_type']:<20}] {c['metadata']['heading'][:40]:<40} sim={c['similarity']:.3f}")
        answer_bad = generate(test["query"], bad_filtered)

        # --- Compare answers ---
        print(f"\n  Answers:")
        print(f"  [Unfiltered] {answer_unfiltered[:150]}...")
        print(f"  [Good]       {answer_good[:150]}...")
        print(f"  [Wrong]      {answer_bad[:150]}...")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. GOOD filters narrow results to the right doc type → better answers")
    print("  2. WRONG filters force retrieval from irrelevant docs → bad answers")
    print("  3. UNFILTERED works OK when the query is specific enough")
    print("  4. Some queries NEED unfiltered search (info scattered across doc types)")
    print("  5. In production: let the LLM CHOOSE the filter (or use a classifier)")
    print("     e.g., 'troubleshoot' → runbooks, 'what happened' → incidents")
    print("  6. Metadata is FREE at index time — always add it, you'll use it later")
    print("=" * 70)
