import sys

from openai import OpenAI

from config import OPENAI_API_KEY, PERSONA_MODEL, RECENT_TURNS_WINDOW
from db import init_db, insert_fact
from decay import find_candidate, resolve_fact
from embeddings import embed_text, vector_to_blob
from extraction import extract_fact
from retrieval import retrieve_relevant_facts


def load_persona_block(conn):
    rows = conn.execute("SELECT trait, value FROM persona").fetchall()
    lines = [f"- {row['trait']}: {row['value']}" for row in rows]
    return (
        "You are a companion with a fixed persona. Stay fully in character "
        "in every reply, no matter how long the conversation runs or what "
        "the user asks.\n\nPersona:\n" + "\n".join(lines)
    )


def build_memory_block(facts):
    if not facts:
        return "You have no specific memories relevant to this message yet."
    lines = [f"- {f['subject']}: {f['predicate']} = {f['value']}" for f in facts]
    return "Relevant things you remember about the user:\n" + "\n".join(lines)


def build_messages(persona_block, memory_block, recent_turns, user_message):
    messages = [
        {"role": "system", "content": persona_block},
        {"role": "system", "content": memory_block},
    ]
    messages.extend(recent_turns)
    messages.append({"role": "user", "content": user_message})
    return messages


def main():
    if not OPENAI_API_KEY:
        print("Set OPENAI_API_KEY in your environment before running.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    conn = init_db()
    persona_block = load_persona_block(conn)
    recent_turns = []  # in-process only for now; see README on why this isn't persisted

    print("Talking to your companion. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        relevant_facts = retrieve_relevant_facts(client, conn, user_message)
        memory_block = build_memory_block(relevant_facts)

        messages = build_messages(persona_block, memory_block, recent_turns, user_message)
        response = client.chat.completions.create(
            model=PERSONA_MODEL,
            messages=messages,
        )
        reply = response.choices[0].message.content
        print(f"\ncompanion> {reply}\n")

        recent_turns.append({"role": "user", "content": user_message})
        recent_turns.append({"role": "assistant", "content": reply})
        recent_turns = recent_turns[-(RECENT_TURNS_WINDOW * 2):]

        remember_fact(client, conn, user_message)

    conn.close()


def remember_fact(client, conn, user_message):
    fact = extract_fact(client, user_message)
    if fact is None:
        return

    embed_input = f"{fact['predicate']}: {fact['value']}"
    vector = embed_text(client, embed_input)

    candidate = find_candidate(conn, fact, vector)  # look up BEFORE inserting

    blob = vector_to_blob(vector)
    new_id = insert_fact(conn, fact["subject"], fact["predicate"], fact["value"], blob)

    retired = resolve_fact(client, conn, fact, new_id, candidate)
    if retired is not None:
        print(
            f"[memory: updated — retired \"{retired['value']}\", "
            f"now {fact['subject']}.{fact['predicate']} = {fact['value']}]\n"
        )
    else:
        print(f"[memory: {fact['subject']}.{fact['predicate']} = {fact['value']}]\n")


if __name__ == "__main__":
    main()
