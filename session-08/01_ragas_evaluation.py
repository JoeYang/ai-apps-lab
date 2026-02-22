"""
Session 8, Task 1: RAG Evaluation with RAGAS
==============================================
Set up RAGAS to evaluate your RAG pipeline with four core metrics:

  1. Faithfulness — does the answer stick to the context?
  2. Answer Relevancy — does the answer address the question?
  3. Context Precision — are relevant docs ranked higher?
  4. Context Recall — did we retrieve all needed information?

We also implement a custom LLM-as-judge evaluator for comparison.

Run: python 01_ragas_evaluation.py

Requires: pip install ragas chromadb openai anthropic datasets
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass

import chromadb
import openai
import anthropic
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

oai = openai.OpenAI()
claude = anthropic.Anthropic()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-5-20250929"
DOCS_DIR = Path(__file__).parent.parent / "session-06" / "docs"


# ============================================================
# RAG pipeline (simplified from Session 7)
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
    return all_chunks


def build_index(chunks: list[Chunk]) -> chromadb.Collection:
    client = chromadb.Client()
    collection = client.create_collection(
        name="ragas_eval", metadata={"hnsw:space": "cosine"}
    )
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


def retrieve(collection: chromadb.Collection, query: str, top_k: int = 3) -> list[str]:
    """Retrieve top-k document texts."""
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    results = collection.query(
        query_embeddings=[resp.data[0].embedding], n_results=top_k
    )
    return results["documents"][0]


def generate_answer(query: str, contexts: list[str]) -> str:
    """Generate an answer from retrieved contexts."""
    context_str = "\n\n---\n\n".join(contexts)
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"Answer the question based ONLY on the provided context. "
                f"If the context doesn't contain the answer, say so.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}"
            ),
        }],
    )
    return response.content[0].text


def run_rag(collection: chromadb.Collection, query: str) -> dict:
    """Run the full RAG pipeline, return all components for evaluation."""
    contexts = retrieve(collection, query, top_k=3)
    answer = generate_answer(query, contexts)
    return {"question": query, "contexts": contexts, "answer": answer}


# ============================================================
# Evaluation dataset — questions with ground-truth answers
# ============================================================

# These are manually created based on the session-06 docs.
# In practice, you'd build a larger set (20+ pairs).
EVAL_SET = [
    {
        "question": "What are the steps for escalating a critical incident?",
        "ground_truth": (
            "For critical incidents, immediately page the on-call engineer, "
            "create a war room channel, notify the team lead within 15 minutes, "
            "and begin the incident response runbook."
        ),
    },
    {
        "question": "How do I restart the matching engine?",
        "ground_truth": (
            "To restart the matching engine, first drain in-flight orders, "
            "verify the order book is persisted, stop the engine process, "
            "apply any configuration changes, and start the engine with "
            "health checks enabled."
        ),
    },
    {
        "question": "What is the max_connections_per_host setting?",
        "ground_truth": (
            "max_connections_per_host is a configuration parameter that limits "
            "the number of concurrent connections to each downstream service. "
            "The default is 100."
        ),
    },
    {
        "question": "What should I do when Kafka broker is down?",
        "ground_truth": (
            "When a Kafka broker is down, check the broker logs for errors, "
            "verify ZooKeeper connectivity, attempt a broker restart, and "
            "if the broker doesn't recover, reassign partitions to healthy brokers."
        ),
    },
    {
        "question": "How do I configure monitoring alerts?",
        "ground_truth": (
            "Configure monitoring alerts by defining alert rules in the "
            "monitoring configuration, setting thresholds for key metrics "
            "like latency and error rate, and configuring notification "
            "channels for the on-call team."
        ),
    },
]


# ============================================================
# RAGAS evaluation
# ============================================================

def run_ragas_evaluation(collection: chromadb.Collection):
    """
    Run RAGAS evaluation on the eval set.

    RAGAS expects a HuggingFace Dataset with columns:
      - question: the query
      - answer: the generated answer
      - contexts: list of retrieved context strings
      - ground_truth: the expected answer (needed for context_recall)
    """
    print("Running RAG pipeline on evaluation set...")
    results = []
    for item in EVAL_SET:
        rag_result = run_rag(collection, item["question"])
        results.append({
            "question": item["question"],
            "answer": rag_result["answer"],
            "contexts": rag_result["contexts"],
            "ground_truth": item["ground_truth"],
        })
        print(f"  Processed: {item['question'][:50]}...")

    # Convert to HuggingFace Dataset (RAGAS requirement)
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    print("\nRunning RAGAS evaluation (this calls an LLM for each metric)...")
    try:
        scores = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        print(f"\n{'='*50}")
        print("RAGAS SCORES")
        print(f"{'='*50}")
        for metric, score in scores.items():
            if isinstance(score, (int, float)):
                bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
                print(f"  {metric:25s}  {score:.4f}  [{bar}]")

        return scores, results

    except Exception as e:
        print(f"\nRAGAS evaluation failed: {e}")
        print("This can happen if RAGAS requires a specific LLM provider setup.")
        print("Falling back to custom LLM-as-judge evaluation...\n")
        return None, results


# ============================================================
# Custom LLM-as-Judge (fallback / complementary approach)
# ============================================================

def llm_judge_evaluate(results: list[dict]):
    """
    Custom evaluation using Claude as a judge.
    This is a simpler, more controllable alternative to RAGAS.
    """
    print(f"\n{'='*50}")
    print("CUSTOM LLM-AS-JUDGE EVALUATION")
    print(f"{'='*50}")

    all_scores = []

    for r in results:
        response = claude.messages.create(
            model=CHAT_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    f"You are evaluating a RAG system. Score each criterion 1-5.\n\n"
                    f"Question: {r['question']}\n\n"
                    f"Retrieved Context:\n{chr(10).join(r['contexts'][:2])}\n\n"
                    f"Generated Answer: {r['answer']}\n\n"
                    f"Ground Truth: {r['ground_truth']}\n\n"
                    f"Score these criteria (1=terrible, 5=perfect):\n"
                    f"1. faithfulness: Does the answer only use info from the context?\n"
                    f"2. relevancy: Does the answer address the question?\n"
                    f"3. completeness: Does the answer cover the key points from ground truth?\n"
                    f"4. context_quality: Did the retriever find the right documents?\n\n"
                    f"Return ONLY a JSON object: "
                    f'{{\"faithfulness\": N, \"relevancy\": N, '
                    f'\"completeness\": N, \"context_quality\": N, '
                    f'\"reasoning\": \"brief explanation\"}}'
                ),
            }],
        )

        text = response.content[0].text.strip()
        try:
            scores = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            scores = json.loads(match.group()) if match else {}

        all_scores.append(scores)
        print(f"\n  Q: {r['question'][:50]}...")
        print(f"    Faithfulness: {scores.get('faithfulness', '?')}/5  "
              f"Relevancy: {scores.get('relevancy', '?')}/5  "
              f"Completeness: {scores.get('completeness', '?')}/5  "
              f"Context: {scores.get('context_quality', '?')}/5")
        print(f"    Reasoning: {scores.get('reasoning', 'n/a')[:80]}")

    # Averages
    metrics = ["faithfulness", "relevancy", "completeness", "context_quality"]
    print(f"\n  {'─'*50}")
    print(f"  AVERAGES (out of 5):")
    for m in metrics:
        vals = [s.get(m, 0) for s in all_scores if isinstance(s.get(m), (int, float))]
        avg = sum(vals) / len(vals) if vals else 0
        bar = "#" * int(avg * 4) + "." * (20 - int(avg * 4))
        print(f"    {m:20s}  {avg:.2f}  [{bar}]")

    return all_scores


# ============================================================
# Main
# ============================================================

def main():
    print("Session 8 — RAG Evaluation with RAGAS\n")

    chunks = load_and_chunk()
    print(f"Loaded {len(chunks)} chunks")
    collection = build_index(chunks)
    print(f"Indexed {collection.count()} chunks\n")

    # Run RAGAS evaluation
    ragas_scores, results = run_ragas_evaluation(collection)

    # Always run custom LLM-as-judge too (for comparison / fallback)
    llm_scores = llm_judge_evaluate(results)

    print(f"\n\n{'='*50}")
    print("KEY TAKEAWAYS")
    print(f"{'='*50}")
    print("""
  1. RAGAS provides standardized metrics that the community uses
     to benchmark RAG systems. Good for comparing approaches.

  2. Faithfulness catches hallucination — the answer inventing
     facts not in the retrieved context.

  3. Context Precision/Recall tell you about retrieval quality
     separately from generation quality.

  4. Custom LLM-as-judge is more flexible — you define the criteria
     that matter for YOUR use case.

  5. Both approaches cost money (LLM calls per evaluation). Budget
     for eval costs in production systems.

  6. Use these metrics to compare: chunking strategies, retrieval
     methods, reranking, prompt templates, models.
""")


if __name__ == "__main__":
    main()
