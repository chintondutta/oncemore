import numpy as np

from config import TOP_K_FACTS
from db import fetch_active_facts
from embeddings import blob_to_vector, embed_text


def _cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_relevant_facts(client, conn, query_text, query_vector=None, k=TOP_K_FACTS):
    """Top-k active facts by cosine similarity to query_text. No similarity
    floor — PRD §6.2 asks for top-k, not a threshold, so with few facts in
    the DB everything currently known is just "the top-k of what exists"."""
    facts = fetch_active_facts(conn)
    if not facts:
        return []

    if query_vector is None:
        query_vector = embed_text(client, query_text)

    scored = [
        (_cosine_similarity(query_vector, blob_to_vector(fact["embedding"])), fact)
        for fact in facts
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [fact for _, fact in scored[:k]]
