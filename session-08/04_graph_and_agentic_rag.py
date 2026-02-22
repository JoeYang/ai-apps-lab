"""
Session 8, Task 4: Graph RAG & Agentic RAG Patterns
=====================================================
Demonstrate two advanced RAG architectures:

  1. Graph RAG — extract entities/relationships, build a knowledge graph,
     traverse it during retrieval
  2. Agentic RAG — an agent that decides when and how to retrieve,
     with multi-step reasoning

Run: python 04_graph_and_agentic_rag.py

Requires: pip install chromadb openai anthropic
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
DOCS_DIR = Path(__file__).parent.parent / "session-06" / "docs"


# ============================================================
# Shared: load documents
# ============================================================

def load_documents() -> dict[str, str]:
    docs = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs[path.name] = path.read_text()
    return docs


# ============================================================
# PART 1: Graph RAG
# ============================================================

@dataclass
class Entity:
    name: str
    entity_type: str  # system, process, config, concept, etc.
    description: str = ""


@dataclass
class Relationship:
    source: str       # entity name
    target: str       # entity name
    relation: str     # depends_on, configures, monitors, etc.
    description: str = ""


@dataclass
class KnowledgeGraph:
    entities: dict[str, Entity] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)

    def add_entity(self, entity: Entity):
        self.entities[entity.name.lower()] = entity

    def add_relationship(self, rel: Relationship):
        self.relationships.append(rel)

    def get_neighbors(self, entity_name: str, max_hops: int = 2) -> set[str]:
        """Get all entities within max_hops of the given entity."""
        name = entity_name.lower()
        visited = {name}
        frontier = {name}

        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for rel in self.relationships:
                    if rel.source.lower() == node and rel.target.lower() not in visited:
                        next_frontier.add(rel.target.lower())
                    elif rel.target.lower() == node and rel.source.lower() not in visited:
                        next_frontier.add(rel.source.lower())
            visited |= next_frontier
            frontier = next_frontier

        return visited

    def get_subgraph_text(self, entity_names: set[str]) -> str:
        """Get a text description of the subgraph around given entities."""
        lines = []
        for name in entity_names:
            if name in self.entities:
                e = self.entities[name]
                lines.append(f"Entity: {e.name} ({e.entity_type}) — {e.description}")

        for rel in self.relationships:
            if rel.source.lower() in entity_names or rel.target.lower() in entity_names:
                lines.append(f"Relationship: {rel.source} —[{rel.relation}]→ {rel.target}")
                if rel.description:
                    lines.append(f"  Detail: {rel.description}")

        return "\n".join(lines)


# --- Entity & relationship extraction ---

def extract_entities_and_relationships(doc_name: str, doc_text: str) -> tuple[list[Entity], list[Relationship]]:
    """Use an LLM to extract entities and relationships from a document."""
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": (
                f"Extract entities and relationships from this technical document.\n\n"
                f"Document ({doc_name}):\n```\n{doc_text[:3000]}\n```\n\n"
                f"Return a JSON object with:\n"
                f"- entities: array of {{\"name\": str, \"type\": str, \"description\": str}}\n"
                f"  Types: system, process, config, concept, tool, team\n"
                f"- relationships: array of {{\"source\": str, \"target\": str, "
                f"\"relation\": str, \"description\": str}}\n"
                f"  Relations: depends_on, configures, monitors, triggers, "
                f"part_of, uses, alerts\n\n"
                f"Extract 5-10 entities and 5-10 relationships. "
                f"Return ONLY the JSON object."
            ),
        }],
    )

    text = response.content[0].text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(match.group()) if match else {"entities": [], "relationships": []}

    entities = [
        Entity(name=e["name"], entity_type=e.get("type", "unknown"),
               description=e.get("description", ""))
        for e in data.get("entities", [])
    ]
    relationships = [
        Relationship(source=r["source"], target=r["target"],
                     relation=r.get("relation", "related_to"),
                     description=r.get("description", ""))
        for r in data.get("relationships", [])
    ]
    return entities, relationships


def build_knowledge_graph(docs: dict[str, str]) -> KnowledgeGraph:
    """Build a knowledge graph from all documents."""
    kg = KnowledgeGraph()

    for doc_name, doc_text in docs.items():
        print(f"  Extracting from {doc_name}...")
        entities, relationships = extract_entities_and_relationships(doc_name, doc_text)
        for e in entities:
            kg.add_entity(e)
        for r in relationships:
            kg.add_relationship(r)

    print(f"  Graph: {len(kg.entities)} entities, {len(kg.relationships)} relationships")
    return kg


def identify_query_entities(kg: KnowledgeGraph, query: str) -> list[str]:
    """Identify which entities in the graph are relevant to the query."""
    entity_list = ", ".join(kg.entities.keys())
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Given this query and list of known entities, which entities "
                f"are relevant? Return ONLY a JSON array of entity names.\n\n"
                f"Query: {query}\n"
                f"Known entities: {entity_list}"
            ),
        }],
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        return json.loads(match.group()) if match else []


def graph_rag_query(kg: KnowledgeGraph, docs: dict[str, str], query: str) -> str:
    """
    Graph RAG: find relevant entities, traverse the graph for context,
    then generate an answer.
    """
    # Step 1: identify relevant entities
    relevant = identify_query_entities(kg, query)
    print(f"    Relevant entities: {relevant}")

    # Step 2: expand to neighbors (1-hop)
    all_entities = set()
    for entity in relevant:
        neighbors = kg.get_neighbors(entity.lower(), max_hops=1)
        all_entities |= neighbors
    print(f"    Expanded to {len(all_entities)} entities (with neighbors)")

    # Step 3: get subgraph context
    graph_context = kg.get_subgraph_text(all_entities)

    # Step 4: also include raw doc text for grounding
    doc_context = "\n\n".join(list(docs.values()))[:2000]

    # Step 5: generate answer
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"Use the knowledge graph AND document context to answer.\n\n"
                f"Knowledge Graph:\n{graph_context}\n\n"
                f"Document Context:\n{doc_context}\n\n"
                f"Question: {query}"
            ),
        }],
    )
    return response.content[0].text


def demo_graph_rag():
    print(f"\n{'='*70}")
    print("PART 1: Graph RAG")
    print(f"{'='*70}")

    docs = load_documents()
    print(f"\nBuilding knowledge graph from {len(docs)} documents...")
    kg = build_knowledge_graph(docs)

    # Show the graph
    print(f"\n  Sample entities:")
    for name, entity in list(kg.entities.items())[:5]:
        print(f"    {entity.name} ({entity.entity_type}): {entity.description[:60]}")
    print(f"\n  Sample relationships:")
    for rel in kg.relationships[:5]:
        print(f"    {rel.source} —[{rel.relation}]→ {rel.target}")

    # Query
    queries = [
        "What systems are affected if Kafka goes down?",
        "How does the matching engine relate to the order book?",
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        answer = graph_rag_query(kg, docs, query)
        print(f"  Answer: {answer[:200]}...")


# ============================================================
# PART 2: Agentic RAG
# ============================================================

def build_vector_index(docs: dict[str, str]) -> chromadb.Collection:
    """Build a simple vector index for the agent to search."""
    client = chromadb.Client()
    collection = client.create_collection(
        name="agentic_rag", metadata={"hnsw:space": "cosine"}
    )

    chunks = []
    for doc_name, doc_text in docs.items():
        sentences = re.split(r'(?<=[.!?])\s+', doc_text)
        current, idx = "", 0
        for sent in sentences:
            if len(current) + len(sent) > 500 and current:
                chunks.append({"text": current.strip(), "source": doc_name, "id": f"{doc_name}::{idx}"})
                idx += 1
                current = ""
            current += sent + " "
        if current.strip():
            chunks.append({"text": current.strip(), "source": doc_name, "id": f"{doc_name}::{idx}"})

    texts = [c["text"] for c in chunks]
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = oai.embeddings.create(input=batch, model=EMBED_MODEL)
        embeddings.extend([e.embedding for e in resp.data])

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection


def agent_search(collection: chromadb.Collection, query: str, top_k: int = 3) -> list[str]:
    """Tool: search documents."""
    resp = oai.embeddings.create(input=[query], model=EMBED_MODEL)
    results = collection.query(query_embeddings=[resp.data[0].embedding], n_results=top_k)
    return results["documents"][0]


def agentic_rag(collection: chromadb.Collection, question: str, max_steps: int = 5) -> str:
    """
    Agentic RAG: the agent decides when and how to search.

    The agent follows a Reason → Act → Observe loop:
    1. Reason about what information is needed
    2. Decide to search (with a specific query) or answer
    3. Observe the search results
    4. Repeat until ready to answer
    """
    messages = [{
        "role": "user",
        "content": (
            f"You are a research agent with access to a document search tool. "
            f"Answer the user's question by searching for relevant information.\n\n"
            f"Available tool:\n"
            f"- search(query) — searches internal documents\n\n"
            f"Process:\n"
            f"1. Think about what you need to find\n"
            f"2. Call search with a specific query\n"
            f"3. Review results and decide if you need more info\n"
            f"4. When ready, provide your final answer\n\n"
            f"Format your response as:\n"
            f"THOUGHT: your reasoning\n"
            f"ACTION: search(\"your query\") OR answer(\"your final answer\")\n\n"
            f"Question: {question}"
        ),
    }]

    all_context = []
    steps = []

    for step in range(max_steps):
        response = claude.messages.create(
            model=CHAT_MODEL,
            max_tokens=400,
            messages=messages,
        )
        text = response.content[0].text
        steps.append(text)

        # Check if agent wants to answer
        answer_match = re.search(r'answer\("(.+?)"\)', text, re.DOTALL)
        if not answer_match:
            answer_match = re.search(r'answer\((.+?)\)', text, re.DOTALL)
        if answer_match:
            final_answer = answer_match.group(1)
            print(f"    Step {step+1}: ANSWER")
            return final_answer

        # Check if agent wants to search
        search_match = re.search(r'search\("(.+?)"\)', text)
        if not search_match:
            search_match = re.search(r'search\((.+?)\)', text)

        if search_match:
            search_query = search_match.group(1).strip('"\'')
            results = agent_search(collection, search_query)
            all_context.extend(results)
            print(f"    Step {step+1}: SEARCH \"{search_query}\" → {len(results)} results")

            # Feed results back to the agent
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n" + "\n---\n".join(results[:3]),
            })
        else:
            # Agent didn't follow format — treat the whole response as the answer
            print(f"    Step {step+1}: (direct response)")
            return text

    # Max steps reached — generate answer from collected context
    context = "\n\n".join(all_context[:5])
    response = claude.messages.create(
        model=CHAT_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return response.content[0].text


def demo_agentic_rag():
    print(f"\n{'='*70}")
    print("PART 2: Agentic RAG")
    print(f"{'='*70}")

    docs = load_documents()
    collection = build_vector_index(docs)
    print(f"  Indexed {collection.count()} chunks\n")

    queries = [
        # Simple — agent should search once and answer
        "What is the incident escalation process?",
        # Complex — agent should search multiple times
        "Compare the monitoring configuration with the incident response procedures. "
        "Are there any gaps where alerts don't have corresponding runbook entries?",
    ]

    for query in queries:
        print(f"\n  Query: \"{query[:70]}...\"" if len(query) > 70 else f"\n  Query: \"{query}\"")
        print(f"  Agent steps:")
        answer = agentic_rag(collection, query)
        print(f"  Answer: {answer[:200]}...")


# ============================================================
# Main
# ============================================================

def main():
    print("Session 8 — Graph RAG & Agentic RAG Patterns\n")

    demo_graph_rag()
    demo_agentic_rag()

    print(f"\n\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  GRAPH RAG:
  1. Extracts entities and relationships → builds a knowledge graph.
  2. At query time, finds relevant entities and traverses neighbors.
  3. Great for "what's connected to X?" and multi-hop questions.
  4. Expensive to build (LLM extraction), but the graph is reusable.
  5. Microsoft's GraphRAG adds community detection for summarization.

  AGENTIC RAG:
  1. An agent decides WHEN and HOW to retrieve (not a fixed pipeline).
  2. Can search multiple times, reformulate queries, combine sources.
  3. Follows Reason → Act → Observe loop (ReAct pattern).
  4. Most powerful but most expensive — multiple LLM + retrieval calls.
  5. You'll build full agents in Phase 3 (Sessions 9-13).

  WHEN TO USE WHAT:
  - Simple lookups        → Standard RAG
  - Better ranking        → + Reranking
  - Relationship queries  → Graph RAG
  - Complex research      → Agentic RAG
""")


if __name__ == "__main__":
    main()
