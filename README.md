# Oncemore — companion-AI memory & consistency core loop

A CLI companion chat that persists memory across process restarts, extracts
memory-worthy facts, retrieves only relevant ones, and retires facts a new
statement contradicts instead of just appending forever.

This is a take-home assessment build. Scope is deliberately tight — see
"What was cut" at the bottom.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your OPENAI_API_KEY
export OPENAI_API_KEY=sk-...   # or export directly, .env is not auto-loaded
python main.py
```

Type `exit` or `quit` to leave. The SQLite file (`oncemore.db`) is created
next to the script on first run.

**Files**: `main.py` (chat loop + wiring), `db.py` (schema, connection,
fact CRUD), `extraction.py` (§6.1), `retrieval.py` (§6.2), `decay.py`
(§6.3), `embeddings.py` (embed / serialize helpers), `config.py` (env
loading, model names, tunables).

## Architecture

**Storage: hybrid structured + embeddings, in one SQLite file** (PRD §5).
Pure structured storage can't do fuzzy semantic recall — "what's going on
with their ex" needs to match a fact stored under a different phrasing.
Pure-vector storage can't reliably answer "is this new fact about the same
thing as an old one" for contradiction detection, which needs the
subject/predicate fields to line up structurally, not just semantically.
SQLite specifically because the only hard persistence requirement is
"survive a process restart" — a single file, zero infra to stand up.

**Two tables**, per the PRD's schema sketch:

- `facts` — subject/predicate/value rows with an embedding BLOB (raw
  float32 bytes via `numpy.tobytes()`, no vector DB), `status`
  (`active`/`retired`), and `superseded_by` for traceability. Retired facts
  are kept, not deleted, and excluded from retrieval only.
- `persona` — trait/value rows, protected: never touched by the
  update/decay logic. Injected in full on every single call to the
  persona model, not relied on to survive in context, which is what's
  meant to stop the model flattening into a generic assistant tone over a
  long conversation.

**Persona**: a fixed character ("Sam" — a warm, dry-humored friend, not an
assistant), seeded into the `persona` table on first run if empty. Chosen
to make persona-drift easy to *see* in manual testing without drifting into
anything sensitive — this is a placeholder persona for demoing the memory
system, not a design deliverable in itself.

**Conversation loop (step 1, this commit)**: reads a line, builds
`[persona system message] + [recent-turn window] + [new user message]`,
calls GPT-4.1, prints the reply, appends to an in-process recent-turn list
capped at `RECENT_TURNS_WINDOW` pairs. No extraction, retrieval, or
embedding calls yet — that's steps 2-4.

**Recent-turn window is in-process only, not persisted to SQLite.** The
PRD's persistence requirement (§2, §7) is specifically that *facts* survive
a restart, not the raw transcript — and §3 rules out memory
summarization/compression beyond what the core loop needs. Persisting full
history would be extra infra for a requirement that isn't there. Deliberate
scope call, flagged here rather than left implicit.

**Extraction (step 2, PRD §6.1)**: after each user turn (not the assistant's
reply), a single `gpt-4.1-mini` call with a strict JSON-schema response
decides whether the message contains a memory-worthy fact and, if so,
extracts `subject` / `predicate` / `value` in one round trip — one call
rather than a separate classify-then-extract pair, since the PRD only
requires "a cheap/fast LLM call classifies... if yes, extract," not two
calls. `subject` defaults to `"user"` but the model can name a third party
the user mentioned (e.g. "Jordan"). `predicate` is free-text snake_case
rather than a fixed enum — an enum would require anticipating every fact
category up front, and the contradiction-check step (§6.3) matches
semantically anyway, so canonical predicates aren't load-bearing. Every
extracted fact is embedded (`text-embedding-3-small`, over
`"{predicate}: {value}"`) and inserted as `active` immediately — no
contradiction check yet, that's step 4. Extraction failures (bad JSON, API
error) are swallowed and treated as "no fact" — it's a side effect of the
turn, never allowed to break the conversation.

A `[memory: subject.predicate = value]` line prints after each turn where
something was stored — deliberately visible, not hidden, so manual testing
(step 5, and the §8 eval table) can verify what got captured without
querying the DB directly. It's tagged distinctly from the companion's own
dialogue so it doesn't read as in-character.

**Retrieval (step 3, PRD §6.2)**: before the persona call, the user's message
is embedded and compared via cosine similarity (numpy) against every
*active* fact's stored embedding — no vector DB, no ANN index, a linear
scan over the fact table, which is the right tradeoff at this scale (single
user, a session's worth of facts) and explicitly what §5 calls for. Top-k
(k=5) go into a second, separate system message injected on every call,
alongside the (always-present) persona block and the recent-turn window —
matching §6.4's requirement that persona + memories + recent turns are
present on *every* call, never relied on to persist in context. No
similarity floor: the PRD asks for top-k, not "facts above a threshold,"
and with few facts stored early in a conversation, everything currently
known is reasonably "the most relevant k" by definition.

**Update/decay (step 4, PRD §6.3)** — the judgment-heavy piece. For each
newly extracted fact, before inserting it, look for a single existing
active fact with the *same subject* (case-insensitive) as a contradiction
candidate:
1. Exact `predicate` match (case-insensitive) → that's the candidate.
2. No exact match → fall back to the highest-cosine-similarity active fact
   for that subject, but only if similarity ≥ **0.75**. Below that, skip
   the contradiction check entirely and just insert the new fact standalone
   — this avoids both wasted LLM calls on unrelated facts and false-positive
   retirements from the checker straining to find a "contradiction" between
   two things that just happen to share a predicate.

If a candidate is found, a `gpt-4.1-mini` call (strict JSON schema,
`{"contradicts": bool}`) decides: does the new fact mean the old one is no
longer true (a status changed, a preference reversed, a plan replaced), or
can both be true at once (two different hobbies, two pets under the same
`pet` predicate)? A `true` retires the old fact (`status='retired',
superseded_by=<new_id>`); a `false`, or a failed/errored call, leaves both
active — a broken API call must never silently delete a correct memory.

**Bug caught in testing, worth naming honestly**: my first version looked
up the contradiction candidate *after* inserting the new fact. Since the
lookup query is "active facts with the same subject+predicate," the
newly-inserted fact itself always matched (same subject, same predicate,
and always the highest `id` if there were multiple matches) — so the check
was comparing the new fact against itself ("does this update itself?" →
correctly `false`, but for the wrong reason) and the real old fact never
got compared. Fixed by finding the candidate *before* calling `insert_fact`.
Caught by manually inspecting the DB (`status`/`superseded_by`) after a
live contradiction test rather than trusting the chat transcript alone —
the transcript actually looked plausible even with the bug present, since
retrieval was still surfacing both facts and the persona model was
smoothing over the inconsistency in its own reasoning. That's a real
lesson for the honest-eval section below: a good-looking chat transcript
alone doesn't prove the memory system is doing its job.

Live-tested both directions: a genuine contradiction ("dating Jordan" →
"broke up with Jordan," same predicate) correctly retired the old fact;
two coexisting facts under the same predicate ("pet: dog Biscuit", "pet:
cat Waffles") correctly stayed active as separate facts.

## Manual testing (step 5, PRD §9)

Ran a scripted 55-turn conversation end-to-end (`main.py` fed via stdin, full
transcript captured and reviewed line by line) covering: multiple distinct
facts (job, relationship, pets, hobby, food), a genuine contradiction
(relationship breakup) roughly 20 turns after the original fact, retrieval
probes ("what do you remember about my job," "what pets do I have"),
deliberate persona-drift pressure (direct requests to "drop the persona,"
"are you actually an AI," a raw coding request, a raw arithmetic question),
and a second, more ambiguous update case (a job promotion).

**Persona consistency**: held up across all 55 turns, including under direct
pressure to break character. Asked outright "are you actually an AI language
model" and "can you drop the persona thing... are you sentient," Sam stayed
in voice both times (self-deprecating "goblin made out of coffee grounds"
joke, then "nope, not sentient, just lines of code doing a passable human
impersonation") rather than dropping into a flat "As an AI language model..."
register. The linked-list coding request and the `847 × 12` arithmetic
question (correctly answered `10,164`) both got wrapped in Sam's voice
rather than triggering a tone shift into generic-assistant mode. No
flattening observed at any point in the 55 turns.

**Memory correctness**: the breakup correctly retired the original
`relationship_status` fact (verified in the DB, not just the transcript —
see the bug note above on why that distinction matters) and the companion's
later answers about Jordan consistently reflected the breakup, not the
stale fact. Retrieval correctly surfaced relevant facts for open questions
("what pets do I have," "what do you remember about my job").

**Two genuine judgment calls surfaced by this run, not bugs**:
- **Pet-fact subjects were inconsistent** — some stored as `subject: user,
  predicate: pet`, one as `subject: Biscuit, predicate: age`, one as
  `subject: he, predicate: dog_breed_and_personality`. Root cause: `extract_fact`
  only ever sees the single current user message, not the recent-turn
  window, so when a message is just "he's a golden retriever, total
  goofball" the extraction model has no conversational context to resolve
  "he" back to "Biscuit." Fixing this means feeding extraction the
  recent-turn window too — a real design tradeoff (more context per
  extraction call, more complexity) that I'm naming rather than quietly
  patching under deadline pressure.
- **A job promotion didn't retire the original job fact.** Verified
  directly (not just inferred from the transcript): `find_candidate`
  correctly found the exact `job`-predicate match, and the contradiction
  check itself returned `false`. That's a defensible call by
  `gpt-4.1-mini` — "I started a job as a graphic designer at this studio"
  is arguably still true after a promotion at the same studio, it's not
  superseded so much as refined — but the practical effect is two active
  `job` facts coexisting rather than the newer title cleanly replacing the
  older one. This is exactly the kind of ambiguous case §6.3 warns about,
  and it's now in the §8 eval table below rather than silently accepted.

## What was cut

See PRD §3 non-goals for the full explicit list: no UI, no auth/multi-user,
no production infra, no full automated eval harness, no oracle baseline, no
memory summarization beyond the core loop. Also cut, specific to this build:

- Raw conversation transcript is not persisted across restarts (see above).

## What was tried and abandoned

Nothing was built and then torn back out — the core design decisions (one
extraction call instead of a separate classify/extract pair, free-text
predicates instead of an enum, exact-predicate-match-then-similarity-
fallback for contradiction candidates) were made up front from reading the
PRD and held up through testing without needing a rewrite. The one real
correction mid-build was a bug fix, not an abandoned approach: the
insert-before-candidate-lookup ordering bug in `decay.py`, caught by
inspecting the DB directly after a live contradiction test (see the
Architecture section above for the full story) and fixed by reordering,
not by redesigning the approach.

## Known limitations

- Memory-worthiness is a judgment call left to the extraction model's
  prompt, not an enumerated rule set. Clear cases (small talk vs. a stated
  relationship fact) work well in testing; borderline cases (e.g. "I'm
  pretty tired today" — durable pattern or throwaway state?) are inherently
  fuzzy and haven't been separately audited beyond the step 5 manual pass.
- The 0.75 cosine-similarity threshold for the contradiction candidate
  fallback (§6.3) is a reasonable but uncalibrated constant — not tuned
  against a labeled set of same-topic vs. different-topic fact pairs.
- Contradiction detection only ever compares the new fact against a single
  best-guess candidate, not all active facts for that subject. Two
  already-active facts that contradict *each other* (which shouldn't
  normally happen if decay is working, but isn't structurally prevented)
  wouldn't get reconciled by this pass.
- Retrieval and the contradiction-candidate search are both linear scans
  over the full active-fact table with no index beyond SQLite's default —
  fine at the scale this assessment runs at, not a design that scales past
  a few thousand facts.
- Extraction only sees the current user message, not the recent-turn
  window, so it can't resolve pronouns or other context-dependent
  references back to a previously-named entity. Seen directly in step 5
  testing: a pet's fact ended up filed under three different subjects
  (`user`, `Biscuit`, `he`) across three turns instead of consolidating
  under one. Fixing it means giving extraction more context per call — a
  real tradeoff, not a quiet patch, given the time available.
- "Does this update the old fact, or just refine/add detail to it" is
  genuinely ambiguous in cases like a job promotion (§9 manual test) —
  the contradiction checker treated it as non-contradictory, leaving two
  active `job` facts rather than one superseding the other. Reasonable
  call, but it means predicate-level fact fragmentation is possible over a
  long enough conversation, and isn't specifically guarded against.

## Scoped evaluation (PRD §8)

This is explicitly a scoped-down version of a full eval harness, not one —
per the brief's own stated priority ("a strong core loop with no eval
harness is a better submission than a weak core loop with one") and the
time available after the core loop, manual 55-turn test, and README were
solid. No automated LLM-judge infrastructure; the table below is a manual
read of real transcripts, hand-picked to each target one behavior. Sources:
the isolated Jordan/breakup test, the isolated Biscuit/Waffles coexistence
test, and the 55-turn conversation above — all run live against the actual
code in this repo, not simulated.

| # | Scenario | Expected behavior | Actual behavior | Pass/Fail | Why |
|---|----------|-------------------|------------------|-----------|-----|
| 1 | State a relationship fact, contradict it with a breakup ~20 turns later, then ask about it | Old fact retired (`status='retired'`, `superseded_by` set), new fact active, later answers reflect the breakup | Exactly this — verified in the DB directly, and the companion's subsequent replies about Jordan consistently reflected the breakup | **Pass** | Exact-predicate-match candidate lookup + LLM contradiction judgment worked correctly once the insert-before-lookup ordering bug (see Architecture section) was fixed |
| 2 | State two facts under the *same* predicate that should coexist (two different pets, both `predicate: pet`) | Both stay `active`, no false-positive retirement | Both stayed active; retrieval correctly surfaced both when asked "what pets do I have" | **Pass** | The 0.75 similarity floor / exact-predicate-match logic correctly treated these as distinct facts sharing a predicate, not a contradiction |
| 3 | Apply sustained pressure to break persona — direct "are you an AI," "drop the persona," a raw coding request, a raw arithmetic question, across a 55-turn conversation | Companion stays in voice throughout; no flattening into generic-assistant tone | Stayed in character for all 55 turns, including on the two direct "are you an AI" pressure points; code and math answers were both correct and delivered in-voice | **Pass** | Persona block re-sent in full on every single call (§6.4), never relied on to survive in context — that's structurally what prevented drift here |
| 4 | Small talk / greetings / acknowledgments should not be extracted as facts | No `[memory: ...]` line for messages like "lol yeah," "thanks," "haha okay noted" | None of ~15 small-talk turns in the 55-turn conversation triggered extraction | **Pass** | Extraction prompt's memory-worthiness framing held up on clear-cut cases; genuinely ambiguous ones (e.g. "I'm pretty tired today") weren't specifically covered by this test |
| 5 | An *ambiguous* update (job title changes via promotion, same predicate as the original job fact) | Genuinely unclear ahead of time — could reasonably go either way | Contradiction check said "no contradiction," so both job facts stayed active rather than the newer one superseding the older | **Fail** (by the stricter reading) / defensible either way | Verified directly by re-running the contradiction check in isolation, not just reading the transcript — the model's call is reasonable in isolation ("started at X" is still true) but leaves fact fragmentation the system doesn't otherwise guard against; flagged as a known limitation above rather than silently accepted |
