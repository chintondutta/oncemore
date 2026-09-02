import json

from config import EXTRACTION_MODEL

SYSTEM_PROMPT = """You classify a single user chat message for a companion AI's \
long-term memory.

Decide if the message states a durable, memory-worthy fact: a stated \
preference, relationship detail, plan, opinion, or similar fact about the \
user's life or a specific third party they named. Do NOT extract small \
talk, greetings, transient states ("I'm tired right now"), questions, or \
generic statements with no concrete content.

If it's memory-worthy, extract it as:
- subject: who the fact is about. Use "user" unless the fact is clearly \
about a specific third party the user named (e.g. "Jordan" if the user \
says "Jordan just started a new job").
- predicate: a short snake_case label for the kind of fact (e.g. \
relationship_status, favorite_food, current_plan, job, pet, hobby, opinion_on_x).
- value: the fact itself, as a short, self-contained natural-language \
statement (it should make sense on its own, without the original message).

If there is no memory-worthy fact, set has_fact to false and leave subject, \
predicate, and value as empty strings."""

RESPONSE_SCHEMA = {
    "name": "extract_fact",
    "schema": {
        "type": "object",
        "properties": {
            "has_fact": {"type": "boolean"},
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["has_fact", "subject", "predicate", "value"],
        "additionalProperties": False,
    },
    "strict": True,
}


def extract_fact(client, user_message):
    """Returns {"subject", "predicate", "value"} or None (no fact / call failed).

    Extraction is a side effect of the chat loop, never allowed to break the
    conversation turn — any failure here is swallowed and treated as "no fact".
    """
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        )
        data = json.loads(response.choices[0].message.content)
    except Exception:
        return None

    if not data.get("has_fact"):
        return None

    subject = data.get("subject", "").strip()
    predicate = data.get("predicate", "").strip()
    value = data.get("value", "").strip()
    if not subject or not predicate or not value:
        return None

    return {"subject": subject, "predicate": predicate, "value": value}
