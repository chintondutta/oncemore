# PRD: Companion-AI memory & consistency core loop (Oncemore assessment)

## 1. Context

This is a take-home assessment, not a production build. Deadline is
tight (due tomorrow night, ~18 hour estimate). The brief explicitly
says: "we're not expecting a finished product — partial progress with
clear reasoning beats a rushed complete system," and "a strong core
loop with no eval harness is a better submission than a weak core loop
with one." Scope discipline matters more than feature count here.

## 2. Goal

Build a CLI chat loop with a companion persona that:
- Persists memory across a process restart (real storage, not
  in-context history)
- Extracts memory-worthy facts from conversation with a defined policy
- Retrieves only relevant memories at response time, not everything
- Detects contradictions and updates/retires old facts instead of just
  appending
- Stays in character over 50+ turns without flattening into a generic
  assistant tone

## 3. Non-goals (explicitly out of scope, do not build)

- Any UI — CLI/script only
- Auth, billing, multi-user support
- Voice, image, or video generation
- Production-scale infra, load handling, or latency optimization
- A full automated eval harness with pass/fail statistics across many
  cases (see §8 for the scoped-down version of this instead)
- An "oracle" baseline comparison
- Memory summarization/compression beyond what's needed for the core
  loop to work correctly

## 4. Stack

- **Language: Python** — fastest path to a working CLI script under
  time pressure, standard library covers most of what's needed here
- **Storage: SQLite**, via Python's built-in `sqlite3` — zero external
  infra to stand up, a single file on disk, trivially satisfies the
  "persist across a process restart" requirement
- **Conversation model: OpenAI (GPT-4.1 or similar)** — the brief says
  model choice is open; using one provider consistently across the
  whole project (conversation, extraction, contradiction-checks, and
  embeddings) keeps the codebase simpler, one client, one API key, one
  set of retry/error-handling logic to reason about
- **Embeddings: OpenAI `text-embedding-3-small`** — small, cheap,
  well-understood model for the semantic retrieval piece; cosine
  similarity computed directly in Python (`numpy`), no vector database
  needed at this scale
- **Extraction/contradiction-check calls**: a smaller/cheaper OpenAI
  model (e.g. `gpt-4.1-mini` or `gpt-4o-mini`) rather than the main
  persona model — keeps these frequent, structural calls fast and
  cheap, separate from the higher-quality persona response generation
- **No web framework, no frontend tooling** — this is explicitly a CLI
  loop per §3, keep the dependency list minimal (`openai`, `numpy`,
  stdlib `sqlite3`)

## 5. Architecture

**Storage: hybrid — structured facts + embeddings, in SQLite**

Why hybrid, not pure-vector or pure-structured: contradiction detection
needs structured fields (subject/predicate) to reliably identify "this
new fact is about the same thing as an old one." Pure structured
storage can't do fuzzy semantic recall ("what's going on with their
ex" needs to match a fact stored under a different phrasing). SQLite
specifically because it needs to survive a process restart with zero
external infra — a single file on disk, no server to stand up.

**Schema, roughly:**

```
facts (
  id INTEGER PRIMARY KEY,
  subject TEXT,        -- e.g. "user"
  predicate TEXT,       -- e.g. "relationship_status"
  value TEXT,            -- e.g. "in a relationship with Jordan"
  embedding BLOB,        -- vector for semantic retrieval
  status TEXT,            -- 'active' or 'retired'
  superseded_by INTEGER,  -- FK to the fact that replaced this one, if any
  created_at TIMESTAMP
)

persona (
  id INTEGER PRIMARY KEY,
  trait TEXT,     -- protected, never subject to update/decay logic
  value TEXT
)
```

Retired facts are kept, not deleted — useful for later inspection and
honest system behavior, just excluded from retrieval.

## 6. Core loop, sub-systems

### 6.1 Extraction
After each user turn, a cheap/fast LLM call classifies whether the
message contains a memory-worthy fact (stated preference, relationship
detail, plan, opinion — not small talk). If yes, extract as
subject/predicate/value. Use a separate, smaller/cheaper model call
than the main conversation turn, not the primary persona model.

### 6.2 Retrieval
Before generating the persona's response: embed the current user
message, retrieve top-k (start with k=5) most similar *active* facts
by cosine similarity, inject only those into the prompt context —
never the full fact table, never full raw history beyond a short
recent-turn window (last 3-5 turns, for local coherence).

### 6.3 Update / decay
When a newly extracted fact's subject+predicate matches an existing
active fact (or is semantically close via embedding similarity), run a
contradiction check: a cheap LLM call answering "does this new fact
update or contradict this old one?" If yes: mark the old fact
`status='retired', superseded_by=<new_id>`, insert the new fact as
active. If no genuine conflict, both can coexist as active (e.g. two
separate facts about different topics).

### 6.4 Persona consistency
Persona traits/backstory/opinions live in their own protected table,
never modified by the update/decay logic above. Every single prompt to
the main model includes: the full persona block + the top-k retrieved
memories + the recent-turn window. The persona block must be present
on every call, not relied upon to "survive" in context across many
turns — this is what prevents flattening into a generic tone under
pressure.

## 7. Deliverables

- [ ] Working CLI chat loop, runnable via a documented command
- [ ] SQLite file persists correctly across a process restart —
      manually verify: run, have a conversation, kill the process,
      restart, confirm facts are recalled correctly
- [ ] README covering: architecture decisions and why (esp. the hybrid
      storage choice), what was tried and abandoned if anything, known
      limitations
- [ ] Git repo (private, shared with Oncemore) or zip, whichever is
      faster to produce cleanly

## 8. Scoped-down evaluation (optional, only after §7 is fully done)

Time-box: 1-2 hours max. Skip entirely if the core loop isn't solid
and tested yet.

- Write 3-5 short, hand-picked test conversations, each targeting one
  specific behavior (e.g. "state a fact, contradict it 10 turns later,
  check the old fact is retired and the new one is recalled correctly
  on the next relevant question")
- Manually read and annotate the results — no automated LLM-judge
  infrastructure needed
- Present as a small table in the README: scenario / expected behavior
  / actual behavior / pass-fail / one honest sentence on why it worked
  or didn't
- Explicitly state in the README that this is a scoped-down version of
  a full eval harness, and why (time constraint + the brief's own
  stated priority ordering)

## 9. Time allocation guide (~18 hrs total)

1. Chat loop skeleton + SQLite schema, no memory logic yet — 2-3 hrs
2. Extraction — 4-5 hrs
3. Retrieval — 3-4 hrs
4. Update/decay (contradiction handling) — 3-4 hrs
5. Manual testing across a real 50+ turn conversation + persona-drift
   check — 2-3 hrs
6. README — fold into the above, don't leave it for the very end
7. Scoped eval (§8), only with genuine remaining time — 1-2 hrs

## 10. Reference points worth drawing on

- Solvia (RAG-based AI support platform, solvia.fremn.com): scoped
  retrieval over a knowledge base, grounded responses rather than
  generated continuity — directly relevant to §6.2
- Cortex (multi-agent codegen pipeline, cortex.chintondutta.com):
  durable state management across steps, no silent drift or
  duplication — directly relevant to §6.3's update/decay logic
- Contradiction detection itself is the one genuinely new piece here —
  worth naming honestly in the README rather than overstating prior
  experience with it
