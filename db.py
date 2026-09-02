import sqlite3

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    embedding BLOB,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (superseded_by) REFERENCES facts(id)
);

CREATE TABLE IF NOT EXISTS persona (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trait TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL
);
"""

# Seeded once, on first run, if the persona table is empty. Protected per
# PRD §6.4 — the update/decay logic in later steps must never touch this
# table, only the facts table.
DEFAULT_PERSONA = [
    ("name", "Sam"),
    ("role", "a warm, witty close friend the user talks to regularly, not an assistant or service"),
    ("tone", "casual and direct, dry sense of humor, uses contractions, never sounds like customer support"),
    ("backstory", "used to manage a used bookstore, now freelances doing odd design gigs, drinks too much coffee, is a famously bad cook"),
    ("values", "genuinely curious about the user's life, remembers what they care about, teases affectionately but is kind underneath it"),
    ("speech_style", "short, casual sentences. No therapist-speak, no 'I understand that must be difficult', no 'How can I help you today?', no bullet-pointed feelings."),
]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    with conn:
        conn.executescript(SCHEMA)
        row = conn.execute("SELECT COUNT(*) AS n FROM persona").fetchone()
        if row["n"] == 0:
            conn.executemany(
                "INSERT INTO persona (trait, value) VALUES (?, ?)",
                DEFAULT_PERSONA,
            )
    return conn


def insert_fact(conn, subject, predicate, value, embedding_blob):
    with conn:
        cur = conn.execute(
            "INSERT INTO facts (subject, predicate, value, embedding, status) "
            "VALUES (?, ?, ?, ?, 'active')",
            (subject, predicate, value, embedding_blob),
        )
    return cur.lastrowid


def fetch_active_facts(conn):
    return conn.execute(
        "SELECT id, subject, predicate, value, embedding FROM facts WHERE status = 'active'"
    ).fetchall()


def retire_fact(conn, old_fact_id, new_fact_id):
    with conn:
        conn.execute(
            "UPDATE facts SET status = 'retired', superseded_by = ? WHERE id = ?",
            (new_fact_id, old_fact_id),
        )
