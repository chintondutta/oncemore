import json

import numpy as np

from config import EXTRACTION_MODEL
from db import fetch_active_facts, retire_fact
from embeddings import blob_to_vector

SIMILARITY_THRESHOLD = 0.75  # uncalibrated but reasonable for text-embedding-3-small

CONTRADICTION_SYSTEM_PROMPT = """You compare two facts held in a companion \
AI's long-term memory, about the same subject and about the same or a \
closely related topic: one already stored (OLD), one just stated (NEW).

Decide whether NEW updates or contradicts OLD — meaning a reasonable \
person would say OLD is no longer true, or has been replaced by NEW (a \
status changed, a preference reversed, a plan was replaced by a newer \
plan).

Decide it does NOT contradict if both facts can be true at the same time \
(e.g. two different hobbies, two favorite things filed under similar \
predicates, unrelated details that happen to be topically close)."""

RESPONSE_SCHEMA = {
    "name": "check_contradiction",
    "schema": {
        "type": "object",
        "properties": {"contradicts": {"type": "boolean"}},
        "required": ["contradicts"],
        "additionalProperties": False,
    },
    "strict": True,
}


def find_candidate(conn, new_fact, new_vector):
    """The single existing active fact (same subject) most likely to be
    "the same thing" as new_fact, or None if nothing qualifies."""
    same_subject = [
        f for f in fetch_active_facts(conn)
        if f["subject"].strip().lower() == new_fact["subject"].strip().lower()
    ]
    if not same_subject:
        return None

    exact = [
        f for f in same_subject
        if f["predicate"].strip().lower() == new_fact["predicate"].strip().lower()
    ]
    if exact:
        return max(exact, key=lambda f: f["id"])  # most recent exact-predicate match

    best_fact, best_score = None, 0.0
    for fact in same_subject:
        vector = blob_to_vector(fact["embedding"])
        score = float(
            np.dot(new_vector, vector) / (np.linalg.norm(new_vector) * np.linalg.norm(vector))
        )
        if score > best_score:
            best_fact, best_score = fact, score

    if best_fact is not None and best_score >= SIMILARITY_THRESHOLD:
        return best_fact
    return None


def check_contradiction(client, old_fact, new_fact):
    prompt = (
        f"OLD ({old_fact['subject']}.{old_fact['predicate']}): {old_fact['value']}\n"
        f"NEW ({new_fact['subject']}.{new_fact['predicate']}): {new_fact['value']}"
    )
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        )
        data = json.loads(response.choices[0].message.content)
        return bool(data.get("contradicts"))
    except Exception:
        return False  # a failed check must never silently retire a correct fact


def resolve_fact(client, conn, new_fact, new_fact_id, candidate):
    """candidate must be looked up via find_candidate() BEFORE the new fact
    is inserted — querying after insertion would let the new fact match
    itself (same subject/predicate, and always the highest id), starving
    out the real old fact it should be compared against.

    Retires the candidate if the contradiction check says so. Returns the
    retired fact's row, or None if nothing was retired."""
    if candidate is None:
        return None

    if check_contradiction(client, candidate, new_fact):
        retire_fact(conn, candidate["id"], new_fact_id)
        return candidate
    return None
