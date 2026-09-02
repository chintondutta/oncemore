import os
from pathlib import Path


def _load_dotenv(path=".env"):
    # Minimal stdlib .env loader — avoids adding python-dotenv as a
    # dependency for one convenience feature. Doesn't override real
    # environment variables that are already set.
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


_load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# One provider, three roles (PRD §4).
PERSONA_MODEL = "gpt-4.1"          # main conversation / persona responses
EXTRACTION_MODEL = "gpt-4.1-mini"  # extraction + contradiction-check calls
EMBEDDING_MODEL = "text-embedding-3-small"

DB_PATH = os.environ.get("ONCEMORE_DB_PATH", "oncemore.db")

RECENT_TURNS_WINDOW = 4  # user+assistant turn pairs kept in-process (PRD §6.2: 3-5 turns)
TOP_K_FACTS = 5          # PRD §6.2
