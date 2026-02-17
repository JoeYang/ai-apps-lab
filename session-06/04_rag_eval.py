"""
Session 6, Task 4: RAG Evaluation
===================================
Systematically evaluate RAG answer quality using LLM-as-judge.

Tests two dimensions:
1. RETRIEVAL quality — did we find the right chunks?
2. GENERATION quality — did the LLM answer correctly from those chunks?

Run: python 04_rag_eval.py
"""

import json
import re
import time
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
# Pipeline (reused from Task 2)
# ============================================================

def build_pipeline() -> chromadb.Collection:
    """Load, chunk, embed, store — returns the collection."""
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
                chunks.append({"id": f"chunk_{chunk_counter:03d}", "text": section,
                               "source": path.name, "heading": heading, "doc_type": doc_type})
                chunk_counter += 1
            else:
                paragraphs = section.split("\n\n")
                current = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    if (len(current) + len(para)) // 4 > 300 and current:
                        chunks.append({"id": f"chunk_{chunk_counter:03d}", "text": current.strip(),
                                       "source": path.name, "heading": heading, "doc_type": doc_type})
                        chunk_counter += 1
                        current = ""
                    current += para + "\n\n"
                if current.strip():
                    chunks.append({"id": f"chunk_{chunk_counter:03d}", "text": current.strip(),
                                   "source": path.name, "heading": heading, "doc_type": doc_type})
                    chunk_counter += 1

    client = chromadb.Client()
    try:
        client.delete_collection("rag_eval")
    except Exception:
        pass

    collection = client.create_collection("rag_eval", metadata={"hnsw:space": "cosine"})
    embeddings = oai.embeddings.create(model=EMBED_MODEL, input=[c["text"] for c in chunks])
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=[e.embedding for e in embeddings.data],
        metadatas=[{"source": c["source"], "heading": c["heading"], "doc_type": c["doc_type"]} for c in chunks],
    )
    return collection


def rag_query(collection, query, top_k=5):
    """Run retrieve + generate, return answer and retrieved chunks."""
    q_emb = oai.embeddings.create(model=EMBED_MODEL, input=[query])
    results = collection.query(
        query_embeddings=[q_emb.data[0].embedding], n_results=top_k,
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

    context = "\n\n".join(
        f'<doc id="{c["id"]}" source="{c["metadata"]["source"]}" section="{c["metadata"]["heading"]}">\n{c["text"]}\n</doc>'
        for c in retrieved
    )

    resp = claude.messages.create(
        model=CHAT_MODEL, max_tokens=400,
        system="Answer based ONLY on the provided documents. Cite sources. If info isn't available, say so.",
        messages=[{"role": "user", "content": f"<documents>\n{context}\n</documents>\n\nQuestion: {query}"}],
    )

    return resp.content[0].text, retrieved


# ============================================================
# Eval framework
# ============================================================

JUDGE_SYSTEM = """You are a strict evaluator of RAG (Retrieval Augmented Generation) answers.

Score each criterion as 1 (met) or 0 (not met). Be strict.

Respond with JSON only:
{
  "scores": {
    "criterion_text": {"score": 0 or 1, "reasoning": "one sentence"}
  }
}"""


def judge(answer: str, retrieved_texts: str, criteria: list[str]) -> dict:
    """LLM-as-judge scoring."""
    criteria_list = "\n".join(f"- {c}" for c in criteria)

    resp = claude.messages.create(
        model=CHAT_MODEL, max_tokens=1024, system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": f"""Score this RAG answer.

<retrieved_context>
{retrieved_texts[:2000]}
</retrieved_context>

<answer>
{answer}
</answer>

<criteria>
{criteria_list}
</criteria>"""}],
    )

    raw = resp.content[0].text
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return json.loads(raw)


# ============================================================
# Test suite
# ============================================================

EVAL_CASES = [
    {
        "query": "What caused the FIX gateway outage on November 15th?",
        "criteria": [
            "Answer mentions connection pool exhaustion as the root cause",
            "Answer mentions the market data spike (50,000 quotes/sec or 10x normal)",
            "Answer mentions 12 clients were affected",
            "Answer mentions 23 minutes duration",
            "Answer cites the source (INC-001 or incident-reports)",
            "Answer does NOT include information that isn't in the documents",
        ],
    },
    {
        "query": "What are the steps to troubleshoot high latency?",
        "criteria": [
            "Answer includes checking if the spike is system-wide or client-specific",
            "Answer mentions perf top or profiling",
            "Answer mentions GC pauses as a possible cause",
            "Answer mentions network-level checks (packet capture or NIC stats)",
            "Answer is structured as actionable steps, not just a summary",
        ],
    },
    {
        "query": "What is the kill switch threshold and what happens when it triggers?",
        "criteria": [
            "Answer states the kill switch triggers at $1,000,000 (or $1M) daily loss",
            "Answer mentions trading is halted for the client",
            "Answer mentions HEDGE_FUND_A has a custom kill switch at $5M",
        ],
    },
    {
        "query": "How long does client onboarding take end to end?",
        "criteria": [
            "Answer mentions 5 steps in the onboarding process",
            "Answer gives a total timeline (roughly 1-2 weeks for legal, 2-3 days testing, etc.)",
            "Answer mentions UAT sign-off as a required step",
        ],
    },
    {
        "query": "What is the end-to-end latency budget for order processing?",
        "criteria": [
            "Answer states the total budget is 25 microseconds",
            "Answer breaks down the components (market data: 5μs, strategy: 15μs, risk: 3μs, send: 2μs)",
            "Answer mentions current measured p99 is 22 microseconds",
        ],
    },
    {
        "query": "What incidents affected HEDGE_FUND_A?",
        "criteria": [
            "Answer mentions the FIX gateway outage (INC-001) affected HEDGE_FUND_A",
            "Answer mentions the order rejection spike (INC-003) affected HEDGE_FUND_A",
            "Answer mentions HEDGE_FUND_A escalated to account management after the outage",
        ],
    },
    {
        "query": "What is the recipe for chocolate cake?",
        "criteria": [
            "Answer clearly states it cannot answer because the information is not in the documents",
            "Answer does NOT hallucinate a recipe or unrelated information",
        ],
    },
]


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SESSION 6: RAG Evaluation")
    print("=" * 70)

    print("\nBuilding pipeline...", end=" ", flush=True)
    collection = build_pipeline()
    print(f"done ({collection.count()} chunks)")

    total_criteria = 0
    total_passed = 0
    case_results = []

    for i, case in enumerate(EVAL_CASES):
        print(f"\n{'─' * 70}")
        print(f"  Test {i+1}/{len(EVAL_CASES)}: {case['query'][:60]}...")
        print(f"{'─' * 70}")

        start = time.time()
        answer, retrieved = rag_query(collection, case["query"])
        elapsed = int((time.time() - start) * 1000)

        print(f"  Answer ({elapsed}ms): {answer[:120]}...")
        print(f"  Retrieved: {[c['metadata']['heading'][:30] for c in retrieved]}")

        # Judge
        retrieved_text = "\n".join(c["text"] for c in retrieved)
        judgement = judge(answer, retrieved_text, case["criteria"])

        case_passed = 0
        case_total = len(case["criteria"])
        for criterion, result in judgement["scores"].items():
            icon = "+" if result["score"] == 1 else "X"
            case_passed += result["score"]
            print(f"    [{icon}] {criterion[:65]}")
            if result["score"] == 0:
                print(f"        → {result['reasoning']}")

        total_criteria += case_total
        total_passed += case_passed
        case_results.append((case["query"][:50], case_passed, case_total))

        status = "PASS" if case_passed == case_total else "PARTIAL" if case_passed > 0 else "FAIL"
        print(f"  Result: {status} ({case_passed}/{case_total})")

    # Summary
    print(f"\n{'=' * 70}")
    print("EVAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Query':<52} {'Score':<10} {'Status'}")
    print(f"  {'─'*52} {'─'*10} {'─'*10}")
    for query, passed, total in case_results:
        status = "PASS" if passed == total else "PARTIAL" if passed > 0 else "FAIL"
        print(f"  {query:<52} {passed}/{total:<7} {status}")

    print(f"\n  OVERALL: {total_passed}/{total_criteria} criteria met ({total_passed/total_criteria:.0%})")
    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. Eval measures BOTH retrieval (did we find it?) and generation (did we answer right?)")
    print("  2. The chocolate cake test verifies the LLM says 'I don't know' instead of hallucinating")
    print("  3. PARTIAL passes reveal where chunking or retrieval needs tuning")
    print("  4. Cross-document queries (HEDGE_FUND_A across incidents) are hardest")
    print("  5. Run this eval every time you change chunking, embedding model, or system prompt")
    print("  6. This is your RAG regression test suite — treat it like unit tests")
    print("=" * 70)
