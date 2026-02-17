"""
Session 6, Task 2: Complete RAG Pipeline
=========================================
End-to-end: load documents → chunk → embed → store → retrieve → generate

This is the core pattern behind ChatGPT + files, Cursor, Perplexity,
and every "chat with your docs" product.

Run: python 02_rag_pipeline.py
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
import openai
import anthropic

oai = openai.OpenAI()
claude = anthropic.Anthropic()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5-20250929"
DOCS_DIR = Path(__file__).parent / "docs"


# ============================================================
# Step 1: LOAD — Read documents from disk
# ============================================================

def load_documents() -> list[dict]:
    """Load markdown documents and extract metadata from filename."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append({
            "filename": path.name,
            "text": path.read_text(),
            "type": path.stem.replace("-", "_"),  # incident_reports, runbooks, etc.
        })
    return docs


# ============================================================
# Step 2: CHUNK — Split documents into searchable pieces
# ============================================================

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    heading: str
    doc_type: str

    @property
    def token_estimate(self) -> int:
        return len(self.text) // 4


def chunk_documents(docs: list[dict], max_tokens: int = 300) -> list[Chunk]:
    """Semantic chunking — split on markdown headings, keep structure."""
    chunks = []
    chunk_counter = 0

    for doc in docs:
        # Split on ## and ### headings
        sections = re.split(r'\n(?=#{2,3}\s)', doc["text"])

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Extract heading
            heading_match = re.match(r'^(#{2,3})\s+(.+?)$', section, re.MULTILINE)
            heading = heading_match.group(2) if heading_match else "Introduction"

            # Sub-chunk if too large
            if len(section) // 4 <= max_tokens:
                chunks.append(Chunk(
                    id=f"chunk_{chunk_counter:03d}",
                    text=section,
                    source=doc["filename"],
                    heading=heading,
                    doc_type=doc["type"],
                ))
                chunk_counter += 1
            else:
                paragraphs = section.split("\n\n")
                current_text = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    if (len(current_text) + len(para)) // 4 > max_tokens and current_text:
                        chunks.append(Chunk(
                            id=f"chunk_{chunk_counter:03d}",
                            text=current_text.strip(),
                            source=doc["filename"],
                            heading=heading,
                            doc_type=doc["type"],
                        ))
                        chunk_counter += 1
                        current_text = ""
                    current_text += para + "\n\n"

                if current_text.strip():
                    chunks.append(Chunk(
                        id=f"chunk_{chunk_counter:03d}",
                        text=current_text.strip(),
                        source=doc["filename"],
                        heading=heading,
                        doc_type=doc["type"],
                    ))
                    chunk_counter += 1

    return chunks


# ============================================================
# Step 3: EMBED — Convert chunks to vectors
# ============================================================

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI."""
    response = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ============================================================
# Step 4: STORE — Index chunks in Chroma
# ============================================================

def build_index(chunks: list[Chunk]) -> chromadb.Collection:
    """Create a Chroma collection and index all chunks."""
    client = chromadb.Client()

    try:
        client.delete_collection("trading_rag")
    except Exception:
        pass

    collection = client.create_collection(
        name="trading_rag",
        metadata={"hnsw:space": "cosine"},
    )

    # Embed all chunks
    embeddings = embed_texts([c.text for c in chunks])

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{
            "source": c.source,
            "heading": c.heading,
            "doc_type": c.doc_type,
            "tokens": c.token_estimate,
        } for c in chunks],
    )

    return collection


# ============================================================
# Step 5: RETRIEVE — Find relevant chunks for a query
# ============================================================

def retrieve(collection: chromadb.Collection, query: str, top_k: int = 5,
             filter_type: str | None = None) -> list[dict]:
    """Retrieve the most relevant chunks for a query."""
    query_embedding = embed_texts([query])

    where_filter = {"doc_type": filter_type} if filter_type else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
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


# ============================================================
# Step 6: GENERATE — Answer using retrieved context
# ============================================================

RAG_SYSTEM_PROMPT = """You are a trading operations assistant. Answer questions based ONLY on the provided documents.

## Rules:
1. ONLY use information from the <documents> section to answer
2. If the answer is not in the documents, say "I don't have enough information to answer that"
3. Cite your sources by referencing the document and section (e.g., "According to INC-001...")
4. Be specific — include numbers, dates, and details from the documents
5. Be concise — answer in 2-4 sentences unless the question requires more detail"""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Generate an answer using Claude with retrieved context."""
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk["metadata"]
        context_parts.append(
            f'<doc id="{chunk["id"]}" source="{meta["source"]}" '
            f'section="{meta["heading"]}" similarity="{chunk["similarity"]:.3f}">\n'
            f'{chunk["text"]}\n</doc>'
        )

    context = "\n\n".join(context_parts)

    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=512,
        system=RAG_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""<documents>
{context}
</documents>

Question: {query}""",
        }],
    )

    return response.content[0].text


# ============================================================
# The full pipeline
# ============================================================

def rag_query(collection: chromadb.Collection, query: str, top_k: int = 5,
              filter_type: str | None = None, verbose: bool = True) -> str:
    """Run the full RAG pipeline: retrieve → generate."""
    # Retrieve
    chunks = retrieve(collection, query, top_k=top_k, filter_type=filter_type)

    if verbose:
        print(f"\n  Retrieved {len(chunks)} chunks:")
        for c in chunks:
            preview = c["text"][:80].replace("\n", " ")
            print(f"    [{c['id']}] ({c['similarity']:.3f}) {c['metadata']['heading']}")
            print(f"      {preview}...")

    # Generate
    answer = generate_answer(query, chunks)

    if verbose:
        print(f"\n  Answer:\n  {answer}")

    return answer


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("SESSION 6: Complete RAG Pipeline")
    print("=" * 65)

    # Step 1: Load
    print("\n--- Step 1: Load Documents ---")
    docs = load_documents()
    for doc in docs:
        print(f"  {doc['filename']}: {len(doc['text']):,} chars")

    # Step 2: Chunk
    print("\n--- Step 2: Chunk Documents ---")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks")
    tokens = [c.token_estimate for c in chunks]
    print(f"  Token range: {min(tokens)}-{max(tokens)} (avg: {sum(tokens)//len(tokens)})")

    # Step 3-4: Embed & Store
    print("\n--- Steps 3-4: Embed & Store in Chroma ---")
    collection = build_index(chunks)
    print(f"  Indexed {collection.count()} chunks")

    # Step 5-6: Retrieve & Generate
    print("\n--- Steps 5-6: Retrieve & Generate ---")

    queries = [
        "What caused the FIX gateway outage and how many clients were affected?",
        "What are the steps to investigate high latency issues?",
        "What are the default risk limits for new clients?",
        "How long does client onboarding take?",
        "What happened with HEDGE_FUND_A during the order rejection incident?",
    ]

    for query in queries:
        print(f"\n{'─' * 65}")
        print(f"  Q: {query}")
        rag_query(collection, query)

    print(f"\n{'=' * 65}")
    print("THE RAG PIPELINE:")
    print("  load docs → chunk (semantic) → embed (OpenAI) → store (Chroma)")
    print("  → retrieve (vector search) → generate (Claude)")
    print()
    print("KEY TAKEAWAYS:")
    print("  1. Chunking quality directly affects answer quality")
    print("  2. The system prompt constrains Claude to ONLY use retrieved docs")
    print("  3. Similarity scores tell you retrieval confidence")
    print("  4. XML tags (<doc>) help Claude distinguish between sources")
    print("  5. This is the same pattern behind ChatGPT + files, Cursor, etc.")
    print("=" * 65)
