"""
Session 8, Task 2: Building an Evaluation Dataset
===================================================
Build a 20+ question-answer evaluation dataset for your RAG system.

Two approaches:
  1. LLM-generated — fast, use Claude to generate Q&A from your docs
  2. Manual curation — review and fix the generated pairs

The output is a JSON file you can reuse across experiments.

Run: python 02_build_eval_dataset.py

Requires: pip install openai anthropic
"""

import json
import re
from pathlib import Path

import anthropic

claude = anthropic.Anthropic()

CHAT_MODEL = "claude-sonnet-4-5-20250929"
DOCS_DIR = Path(__file__).parent.parent / "session-06" / "docs"
OUTPUT_FILE = Path(__file__).parent / "eval_dataset.json"


# ============================================================
# Step 1: Read all documents
# ============================================================

def load_documents() -> dict[str, str]:
    """Load all docs, keyed by filename."""
    docs = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs[path.name] = path.read_text()
    return docs


# ============================================================
# Step 2: LLM-generated Q&A pairs
# ============================================================

def generate_qa_pairs(doc_name: str, doc_text: str, n: int = 5) -> list[dict]:
    """
    Ask Claude to generate Q&A pairs from a document.

    We ask for different difficulty levels and question types
    to get good coverage.
    """
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": (
                f"You are building an evaluation dataset for a RAG system. "
                f"Given the document below, generate {n} question-answer pairs.\n\n"
                f"Requirements:\n"
                f"- Mix of question types: factual lookup, how-to, conceptual\n"
                f"- Mix of difficulty: 2 easy, 2 medium, 1 hard (requires "
                f"combining info from different parts of the doc)\n"
                f"- Answers should be 1-3 sentences, factual, based only on "
                f"the document\n"
                f"- Include the specific section/line the answer comes from\n\n"
                f"Document ({doc_name}):\n"
                f"```\n{doc_text}\n```\n\n"
                f"Return a JSON array of objects with these fields:\n"
                f"- question: the question\n"
                f"- ground_truth: the correct answer\n"
                f"- source: the document filename\n"
                f"- difficulty: easy/medium/hard\n"
                f"- type: factual/how-to/conceptual\n\n"
                f"Return ONLY the JSON array."
            ),
        }],
    )

    text = response.content[0].text.strip()
    try:
        pairs = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        pairs = json.loads(match.group()) if match else []

    return pairs


def generate_negative_examples(docs: dict[str, str], n: int = 5) -> list[dict]:
    """
    Generate questions that CANNOT be answered from the docs.
    A good RAG system should say "I don't know" for these.
    """
    doc_summaries = "\n".join(
        f"- {name}: {text[:200]}..." for name, text in docs.items()
    )

    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": (
                f"Given these document summaries, generate {n} questions that "
                f"CANNOT be answered from these documents. The questions should "
                f"sound plausible (like a real user might ask them) but the "
                f"answers are simply not in the docs.\n\n"
                f"Documents:\n{doc_summaries}\n\n"
                f"Return a JSON array of objects:\n"
                f"- question: the unanswerable question\n"
                f"- ground_truth: \"This information is not available in the documents.\"\n"
                f"- source: \"none\"\n"
                f"- difficulty: \"negative\"\n"
                f"- type: \"unanswerable\"\n\n"
                f"Return ONLY the JSON array."
            ),
        }],
    )

    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        return json.loads(match.group()) if match else []


# ============================================================
# Step 3: Quality checks on the generated dataset
# ============================================================

def validate_dataset(dataset: list[dict]) -> dict:
    """Run basic quality checks on the eval dataset."""
    checks = {
        "total_pairs": len(dataset),
        "with_question": sum(1 for d in dataset if d.get("question")),
        "with_ground_truth": sum(1 for d in dataset if d.get("ground_truth")),
        "with_source": sum(1 for d in dataset if d.get("source")),
        "difficulty_distribution": {},
        "type_distribution": {},
        "avg_question_length": 0,
        "avg_answer_length": 0,
    }

    for d in dataset:
        diff = d.get("difficulty", "unknown")
        checks["difficulty_distribution"][diff] = (
            checks["difficulty_distribution"].get(diff, 0) + 1
        )
        qtype = d.get("type", "unknown")
        checks["type_distribution"][qtype] = (
            checks["type_distribution"].get(qtype, 0) + 1
        )

    questions = [d.get("question", "") for d in dataset]
    answers = [d.get("ground_truth", "") for d in dataset]
    checks["avg_question_length"] = (
        sum(len(q) for q in questions) / len(questions) if questions else 0
    )
    checks["avg_answer_length"] = (
        sum(len(a) for a in answers) / len(answers) if answers else 0
    )

    return checks


# ============================================================
# Step 4: Deduplicate similar questions
# ============================================================

def deduplicate(dataset: list[dict]) -> list[dict]:
    """
    Remove near-duplicate questions using simple overlap heuristic.
    In production, you'd use embedding similarity for this.
    """
    seen = []
    unique = []

    for item in dataset:
        q = item.get("question", "").lower()
        q_words = set(q.split())

        is_dup = False
        for seen_words in seen:
            overlap = len(q_words & seen_words) / max(len(q_words | seen_words), 1)
            if overlap > 0.7:
                is_dup = True
                break

        if not is_dup:
            seen.append(q_words)
            unique.append(item)

    removed = len(dataset) - len(unique)
    if removed:
        print(f"  Removed {removed} near-duplicate questions")
    return unique


# ============================================================
# Main
# ============================================================

def main():
    print("Session 8 — Building an Evaluation Dataset\n")

    # Load documents
    docs = load_documents()
    print(f"Loaded {len(docs)} documents: {', '.join(docs.keys())}\n")

    # Generate Q&A pairs from each document
    all_pairs = []
    for doc_name, doc_text in docs.items():
        print(f"Generating Q&A pairs from {doc_name}...")
        pairs = generate_qa_pairs(doc_name, doc_text, n=7)
        all_pairs.extend(pairs)
        print(f"  Generated {len(pairs)} pairs")

    # Generate negative examples
    print(f"\nGenerating negative (unanswerable) examples...")
    negatives = generate_negative_examples(docs, n=5)
    all_pairs.extend(negatives)
    print(f"  Generated {len(negatives)} negative examples")

    # Deduplicate
    print(f"\nDeduplicating...")
    all_pairs = deduplicate(all_pairs)

    # Validate
    checks = validate_dataset(all_pairs)
    print(f"\n{'='*50}")
    print("DATASET QUALITY CHECKS")
    print(f"{'='*50}")
    print(f"  Total pairs:          {checks['total_pairs']}")
    print(f"  With question:        {checks['with_question']}")
    print(f"  With ground truth:    {checks['with_ground_truth']}")
    print(f"  Avg question length:  {checks['avg_question_length']:.0f} chars")
    print(f"  Avg answer length:    {checks['avg_answer_length']:.0f} chars")
    print(f"\n  Difficulty distribution:")
    for diff, count in sorted(checks["difficulty_distribution"].items()):
        print(f"    {diff:12s}: {count}")
    print(f"\n  Question type distribution:")
    for qtype, count in sorted(checks["type_distribution"].items()):
        print(f"    {qtype:15s}: {count}")

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"\nSaved {len(all_pairs)} pairs to {OUTPUT_FILE}")

    # Show a few examples
    print(f"\n{'='*50}")
    print("SAMPLE PAIRS")
    print(f"{'='*50}")
    for pair in all_pairs[:3]:
        print(f"\n  [{pair.get('difficulty', '?')}] [{pair.get('type', '?')}]")
        print(f"  Q: {pair['question']}")
        print(f"  A: {pair['ground_truth'][:100]}...")

    if negatives:
        print(f"\n  [NEGATIVE EXAMPLE]")
        print(f"  Q: {negatives[0].get('question', 'n/a')}")
        print(f"  A: {negatives[0].get('ground_truth', 'n/a')}")

    print(f"\n\n{'='*50}")
    print("NEXT STEPS")
    print(f"{'='*50}")
    print("""
  1. Review eval_dataset.json manually — fix bad questions/answers.
  2. Ensure ground truth answers match what's actually in the docs.
  3. Add more domain-specific questions your real users would ask.
  4. Use this dataset in 01_ragas_evaluation.py and 03_iterate_tune.py
     to benchmark your pipeline improvements.
""")


if __name__ == "__main__":
    main()
