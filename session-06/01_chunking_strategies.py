"""
Session 6, Task 1: Chunking Strategies
========================================
Compare three chunking approaches on the same documents:

1. Fixed-size — split every N characters (dumb but simple)
2. Sentence-aware — split on sentence boundaries
3. Semantic / section-aware — split on document structure (headings, sections)

Run: python 01_chunking_strategies.py
"""

import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    source: str         # filename
    chunk_id: int       # chunk index within the source
    strategy: str       # which chunking strategy produced this
    metadata: dict      # extra info (section heading, etc.)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars for English)."""
        return len(self.text) // 4


DOCS_DIR = Path(__file__).parent / "docs"


def load_documents() -> dict[str, str]:
    """Load all markdown documents."""
    docs = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs[path.name] = path.read_text()
    return docs


# ============================================================
# Strategy 1: Fixed-size chunking
# ============================================================

def chunk_fixed_size(text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Split text into fixed-size chunks with overlap.

    Simple but dumb — may split mid-sentence or mid-word.
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_id=chunk_id,
                strategy="fixed_size",
                metadata={"char_start": start, "char_end": end},
            ))
            chunk_id += 1

        start = end - overlap  # Overlap to avoid splitting context

    return chunks


# ============================================================
# Strategy 2: Sentence-aware chunking
# ============================================================

def split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple regex approach)."""
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentence_aware(text: str, source: str, max_tokens: int = 150, overlap_sentences: int = 1) -> list[Chunk]:
    """Split on sentence boundaries, grouping sentences until max_tokens.

    Respects sentence boundaries — never splits mid-sentence.
    """
    sentences = split_sentences(text)
    chunks = []
    chunk_id = 0
    i = 0

    while i < len(sentences):
        current_sentences = []
        current_tokens = 0

        # Accumulate sentences until we hit the token limit
        j = i
        while j < len(sentences):
            sentence_tokens = len(sentences[j]) // 4
            if current_tokens + sentence_tokens > max_tokens and current_sentences:
                break
            current_sentences.append(sentences[j])
            current_tokens += sentence_tokens
            j += 1

        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_id=chunk_id,
                strategy="sentence_aware",
                metadata={"sentence_start": i, "sentence_end": j},
            ))
            chunk_id += 1

        # Move forward, but overlap by N sentences
        i = max(j - overlap_sentences, i + 1)

    return chunks


# ============================================================
# Strategy 3: Semantic / section-aware chunking
# ============================================================

def chunk_semantic(text: str, source: str, max_tokens: int = 300) -> list[Chunk]:
    """Split on document structure (markdown headings).

    Uses the document's own structure — headings, sections, etc.
    Each section becomes a chunk. Large sections are sub-chunked.
    """
    chunks = []
    chunk_id = 0

    # Split on markdown headings (## or ###)
    sections = re.split(r'\n(?=#{2,3}\s)', text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading if present
        heading_match = re.match(r'^(#{2,3})\s+(.+?)$', section, re.MULTILINE)
        heading = heading_match.group(2) if heading_match else "Introduction"

        # If section is small enough, keep it as one chunk
        section_tokens = len(section) // 4
        if section_tokens <= max_tokens:
            chunks.append(Chunk(
                text=section,
                source=source,
                chunk_id=chunk_id,
                strategy="semantic",
                metadata={"heading": heading},
            ))
            chunk_id += 1
        else:
            # Sub-chunk large sections by paragraph
            paragraphs = section.split("\n\n")
            current_text = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if (len(current_text) + len(para)) // 4 > max_tokens and current_text:
                    chunks.append(Chunk(
                        text=current_text.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        strategy="semantic",
                        metadata={"heading": heading},
                    ))
                    chunk_id += 1
                    current_text = ""
                current_text += para + "\n\n"

            if current_text.strip():
                chunks.append(Chunk(
                    text=current_text.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    strategy="semantic",
                    metadata={"heading": heading},
                ))
                chunk_id += 1

    return chunks


# ============================================================
# Comparison
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("SESSION 6: Chunking Strategies Comparison")
    print("=" * 65)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents:")
    for name, text in docs.items():
        print(f"  {name}: {len(text):,} chars, ~{len(text)//4:,} tokens")

    all_text = "\n\n".join(docs.values())

    strategies = {
        "Fixed-size (500 char, 100 overlap)": lambda: [
            c for name, text in docs.items()
            for c in chunk_fixed_size(text, name)
        ],
        "Sentence-aware (150 tokens)": lambda: [
            c for name, text in docs.items()
            for c in chunk_sentence_aware(text, name)
        ],
        "Semantic / section-aware (300 tokens)": lambda: [
            c for name, text in docs.items()
            for c in chunk_semantic(text, name)
        ],
    }

    for strategy_name, chunk_fn in strategies.items():
        chunks = chunk_fn()
        print(f"\n{'─' * 65}")
        print(f"  Strategy: {strategy_name}")
        print(f"  Chunks: {len(chunks)}")
        token_counts = [c.token_estimate for c in chunks]
        print(f"  Token range: {min(token_counts)} — {max(token_counts)} (avg: {sum(token_counts)//len(token_counts)})")
        print(f"{'─' * 65}")

        # Show first 3 chunks
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk.text[:120].replace("\n", " ")
            print(f"  [{i}] ({chunk.token_estimate} tokens) {preview}...")
            if chunk.metadata.get("heading"):
                print(f"       Section: {chunk.metadata['heading']}")

        # Show a problematic example for fixed-size
        if "Fixed" in strategy_name:
            # Find a chunk that was clearly split mid-sentence
            for chunk in chunks:
                if chunk.text and not chunk.text[-1] in ".!?:\n" and len(chunk.text) > 100:
                    print(f"\n  ** PROBLEM: Chunk {chunk.chunk_id} ends mid-sentence:")
                    print(f"     \"...{chunk.text[-80:]}\"")
                    break

    print(f"\n{'=' * 65}")
    print("COMPARISON:")
    print()
    print(f"  {'Strategy':<35} {'Chunks':<10} {'Pros':<30} {'Cons'}")
    print(f"  {'─'*35} {'─'*10} {'─'*30} {'─'*30}")
    print(f"  {'Fixed-size':<35} {'Many':<10} {'Simple, predictable size':<30} {'Splits mid-sentence'}")
    print(f"  {'Sentence-aware':<35} {'Medium':<10} {'Clean boundaries':<30} {'Ignores document structure'}")
    print(f"  {'Semantic (section-aware)':<35} {'Fewer':<10} {'Preserves context & structure':<30} {'Depends on doc format'}")
    print()
    print("  For trading docs (structured reports, runbooks): SEMANTIC is best")
    print("  → Incident sections stay together, runbook steps aren't split")
    print("  → Heading metadata enables filtering ('show me only runbooks')")
    print("=" * 65)
