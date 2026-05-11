"""
RAG Retriever — Swiss Consumer Data
Uses sentence-transformers for semantic search.
"""

import sys
from pathlib import Path
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from chunks import CHUNKS

# ── Embedding Function ────────────────────────────────────────────────────────

_model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
    def __call__(self, input: Documents) -> Embeddings:
        return _model.encode(list(input)).tolist()

EMBEDDING_FN = SentenceTransformerEmbeddingFunction()
_collection  = None

# ── Collection ────────────────────────────────────────────────────────────────

def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    client      = chromadb.EphemeralClient()
    _collection = client.create_collection(
        name="swiss_consumer_data",
        embedding_function=EMBEDDING_FN,
        metadata={"hnsw:space": "cosine"},
    )
    _collection.add(
        ids       = [c["id"]   for c in CHUNKS],
        documents = [c["text"] for c in CHUNKS],
        metadatas = [
            {
                "source":   c["source"],
                "topic":    c["topic"],
                "personas": ",".join(c["personas"]),
            }
            for c in CHUNKS
        ],
    )
    return _collection

# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve_context(persona: str, question: str, n_results: int = 4) -> str:
    collection = _get_collection()
    query      = f"{persona} Swiss consumer: {question}"
    results    = collection.query(query_texts=[query], n_results=min(len(CHUNKS), 20))

    all_docs  = results["documents"][0]
    all_metas = results["metadatas"][0]

    filtered_docs, filtered_metas = [], []
    for doc, meta in zip(all_docs, all_metas):
        p = meta.get("personas", "")
        if persona in p or "all" in p:
            filtered_docs.append(doc)
            filtered_metas.append(meta)
        if len(filtered_docs) >= n_results:
            break

    if not filtered_docs:
        filtered_docs  = all_docs[:n_results]
        filtered_metas = all_metas[:n_results]

    lines = ["## Relevant Swiss Consumer Data (use this to ground your answer):"]
    for doc, meta in zip(filtered_docs, filtered_metas):
        lines.append(f"- [{meta['source']}] {doc}")
    return "\n".join(lines)

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("family",       "Would you pay CHF 8/month for a Visa card with no foreign fees?"),
        ("retired",      "Would you ever switch from cash to a credit card?"),
        ("student",      "You use Twint for everything. What would make you switch to Visa?"),
        ("professional", "Which Visa card features matter most for your travel spending?"),
        ("mittelstand",  "Would you pay CHF 8/month for a card with 1% cashback on groceries?"),
    ]
    for persona, question in tests:
        print(f"\n=== {persona.upper()} ===")
        print(retrieve_context(persona, question))