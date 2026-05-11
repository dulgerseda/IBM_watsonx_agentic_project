"""
RAG Retriever Tool for IBM watsonx Orchestrate.
Retrieves relevant Swiss consumer data using keyword search (no chromadb).
"""

import sys
from pathlib import Path
from ibm_watsonx_orchestrate.agent_builder.tools import tool

sys.path.insert(0, str(Path(__file__).parent))

@tool
def retrieve_context_for_persona(persona: str, question: str) -> str:
    """
    Retrieves relevant Swiss consumer data chunks for a given persona and question.

    Args:
        persona: The persona type. Must be one of: student, professional, family, retired, mittelstand.
        question: The survey question to retrieve context for.

    Returns:
        Relevant Swiss consumer data as a formatted string to ground the persona response.
    """
    valid_personas = {"student", "professional", "family", "retired", "mittelstand"}
    persona = persona.strip().lower()

    if not persona:
        return "Error: Missing required parameter 'persona'."
    if persona not in valid_personas:
        return f"Error: Invalid persona '{persona}'. Must be one of: {sorted(valid_personas)}"
    if not question or not question.strip():
        return "Error: Missing required parameter 'question'."

    try:
        from chunks import CHUNKS

        # Keyword search — no chromadb needed
        question_words = set(question.lower().split())
        scored = []

        for chunk in CHUNKS:
            personas = chunk.get("personas", [])
            if persona not in personas and "all" not in personas:
                continue
            text = chunk.get("text", "").lower()
            score = sum(1 for word in question_words if word in text)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:4] if scored else []

        if not top:
            # fallback: return all chunks for this persona
            top = [(0, c) for c in CHUNKS if persona in c.get("personas", []) or "all" in c.get("personas", [])][:4]

        lines = ["## Relevant Swiss Consumer Data (use this to ground your answer):"]
        for _, chunk in top:
            lines.append(f"- [{chunk['source']}] {chunk['text']}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error during retrieval: {str(e)}"
