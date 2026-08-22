# GitHub Repo Q&A Assistant

Point it at a repo, a PDF, a CSV, or a live sensor feed, ask questions in plain
English, get answers with citations back to the exact source — no
hallucinated APIs, no "I looked at everything and guessed."

```
> How does hybrid search combine dense and BM25 scores?

Hybrid search normalizes both score lists to [0,1] with min-max scaling,
then combines them as `dense_weight * dense + bm25_weight * bm25`
[src/retrieve/hybrid_search.py:L18-L21].

Citations:
  src/retrieve/hybrid_search.py:L18-L21
```

Gemini for generation, `sentence-transformers` for embeddings and
reranking, Qdrant for vector storage.

---

## Configuration

Everything lives in a `.env` file in the repo root (gitignored — never commit
it). **Two lines are required**, everything else has a working default:

```bash
echo 'LLM_API_KEY=your-key-here' >> .env
echo "JWT_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" >> .env
```

`LLM_API_KEY` is a key for any OpenAI-compatible provider — a free
[Google AI Studio key](https://aistudio.google.com/apikey) needs no credit
card. `JWT_SECRET` signs login sessions (see `src/auth.py`); Docker Compose
refuses to start the backend without both — there's no safe default to ship
for either one.

Copy the block below into `.env` and change what you need:

```bash
# ── Generation ────────────────────────────────────────────────────────────────
LLM_API_KEY=your-key-here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-flash-latest
LLM_BULK_MODEL=gemini-flash-lite-latest

# ── Auth & connector secrets ────────────────────────────────────────────────────
JWT_SECRET=                     # required — see command above
CONNECTOR_ENCRYPTION_KEY=       # required only if you add a Postgres/Redis connector

# ── Local models (downloaded on first run, cached after) ──────────────────────
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
PRELOAD_MODELS=true

# ── Storage ───────────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
REPO_CLONE_DIR=./data/repos
UPLOAD_DIR=./data/uploads
DB_PATH=./data/app.db
HISTORY_TURNS=3

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_FILES=8
HYBRID_CANDIDATE_K=12
HYBRID_DENSE_WEIGHT=1.0
HYBRID_BM25_WEIGHT=0.25
RERANK_TOP_K=6
RERANK_MIN_TOP_SCORE=0.01
```

### Generation

| Setting | Default | What it does |
|---|---|---|
| `LLM_API_KEY` | *(none)* | **Required.** Leave blank only for Ollama, which needs no auth. |
| `LLM_BASE_URL` | Gemini | Any OpenAI-compatible `/chat/completions` host — see the provider table in Quickstart. |
| `LLM_MODEL` | `gemini-flash-latest` | Answers questions, compresses chunks, classifies/decomposes/rewrites questions. |
| `LLM_BULK_MODEL` | `gemini-flash-lite-latest` | Writes per-file/per-chunk ingest summaries and claim-attribution checks. Runs **hundreds of times** per ingest, so use the cheapest model that can write a sentence. |

### Auth & connector secrets

| Setting | Default | What it does |
|---|---|---|
| `JWT_SECRET` | *(none)* | **Required.** Signs login session tokens (see `src/auth.py`). Generate with `python3 -c "import secrets; print(secrets.token_hex(32))"`. Rotating it logs every user out. |
| `CONNECTOR_ENCRYPTION_KEY` | *(none)* | Encrypts Postgres/Redis connector passwords at rest (see `src/crypto.py`) — required only if a space adds a live-data or historical-report connector. Generate with `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. |

### Local models

| Setting | Default | What it does |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Turns text into vectors, for both indexing and querying. Changing it invalidates every stored vector — re-ingest after. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder that re-scores candidates. The slowest stage on CPU (~1s per candidate); a smaller model like `cross-encoder/ms-marco-MiniLM-L-6-v2` is ~6× faster but emits raw logits, so `RERANK_MIN_TOP_SCORE` needs re-deriving. |
| `PRELOAD_MODELS` | `true` | Load both models at startup (~15s) rather than charging it to the first question. Set `false` in development so `--reload` doesn't pay it on every edit. |

### Storage

| Setting | Default | What it does |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Vector store. Compose overrides this to `http://qdrant:6333` inside the container network. |
| `REPO_CLONE_DIR` | `./data/repos` | Where repos are shallow-cloned. |
| `UPLOAD_DIR` | `./data/uploads` | Where uploaded PDF/DOCX/CSV files are stored. |
| `DB_PATH` | `./data/app.db` | SQLite file holding users, spaces, sources, connectors, chats, messages, and the Q&A cache. |
| `HISTORY_TURNS` | `3` | How many prior user+assistant turns are sent to the LLM as chat context. |

### Retrieval

| Setting | Default | What it does |
|---|---|---|
| `TOP_FILES` | `8` | Files the router shortlists before any chunk search runs. |
| `HYBRID_CANDIDATE_K` | `12` | Candidates passed to reranking. **This sets query latency** — reranking costs ~1s per candidate on CPU; the final top-k has consistently come from fused positions ≤16. |
| `HYBRID_DENSE_WEIGHT` | `1.0` | Weight on cosine similarity in the fusion. |
| `HYBRID_BM25_WEIGHT` | `0.25` | Weight on keyword overlap. The 4:1 ratio mirrors Anthropic's contextual-retrieval finding. |
| `RERANK_TOP_K` | `6` | Chunks that reach compression and the final prompt. |
| `RERANK_MIN_TOP_SCORE` | `0.01` | If the *best* reranked chunk scores below this, nothing in the space answers the question and the query short-circuits without calling the LLM. Gates the top score rather than each chunk's, because supporting context legitimately scores low. |

A handful of other knobs (semantic-cache threshold, sub-question fan-out,
sufficiency/route gates, wide-answer token budgets) exist in `src/config.py`
with tuning notes inline — the table above covers what you're likely to
actually touch.

---

## What this is

A multi-user workspace for asking questions over your own data, not just a
single repo:

- **Accounts & roles.** Email/password auth (JWT session cookie); the first
  account created becomes admin automatically. Admins manage every user's
  role and which spaces they can see; a regular user only sees spaces
  they've been assigned to.
- **Spaces, mixing source types.** Each space is an isolated collection of
  sources — a cloned repo, uploaded PDFs/DOCX/CSVs, or pasted text — queried
  together. Admin-only to create/manage; any assigned user can chat.
- **Live-data & historical-report tool-calling.** A space can add a
  Redis or Postgres connector. A question phrased as a *current* reading
  ("what's the pressure at X right now") is answered by fetching live Redis
  keys directly; a question about a past window ("pressure report for
  yesterday", "flow trend last week") resolves a date range and metric
  granularity and answers from Postgres aggregate tables — both skip the
  normal chunk-retrieval path entirely and answer from real fetched data.
- **A real chat UI**, not just a trace viewer — SSE-streamed answers, charts
  and tables when the question calls for one, per-chat and per-message
  history, cancellable in-flight generation.
- **Insights dashboards** (admin-only): per-space retrieval/cache/latency/
  token/faithfulness metrics over a date range, a per-chat question
  breakdown down to the individual pipeline trace, and a per-user activity
  view for token/question/cache stats across every space they've used.
- **Claim-level faithfulness checking**, computed after generation rather
  than self-reported by the model: every factual claim in an answer —
  including a live-data table's numeric readings, not just its prose — is
  attributed back to the source chunk (or Redis/Postgres reading) that
  actually supports it. Unsupported claims are flagged in the trace and
  roll up into the faithfulness rate on the insights dashboards.

The rest of this README covers the retrieval engine underneath all of that —
the part that hasn't changed shape since the original single-repo version.

---

## Why this exists

Feeding an LLM "the whole repo" doesn't scale past a few dozen files, and
naive fixed-size-chunk RAG breaks on code specifically:

| Problem with naive RAG on code | How this pipeline fixes it |
|---|---|
| Splitting every N characters cuts a function in half | **AST-aware chunking** (tree-sitter) — split on function/class boundaries, not char counts |
| A chunk like `def verify(token): ...` has no idea what file or module it's from | **Contextual headers** — an LLM-written "where this fits" sentence is embedded alongside the code |
| Searching all chunks at once returns noisy matches from unrelated files | **Hierarchical routing** — first pick relevant *files* from summaries, then search chunks only inside them |
| Top-k cosine matches are approximately right but bloated with irrelevant lines | **Cross-encoder rerank + line-level compression** — re-score precisely, then strip non-answering lines before they hit the prompt |

The result: 5-8 small, self-describing, pre-trimmed chunks in the prompt
instead of a wall of raw source — better answers, smaller context, cheaper
calls.

---

## Architecture

```
INGEST  (add a source to a space, once per source)
┌──────────┐   ┌────────────┐   ┌─────────────┐   ┌────────────────┐   ┌─────────┐
│ clone/   │ → │ walk/parse │ → │ AST-chunk   │ → │ summarize file  │ → │ embed + │
│ upload   │   │ (ext/size  │   │ (tree-sitter│   │ + repo (LLM),   │   │ upsert  │
│          │   │  filtered) │   │  per lang)  │   │ contextualize   │   │ to      │
│          │   │            │   │             │   │ each chunk (LLM)│   │ Qdrant  │
└──────────┘   └────────────┘   └─────────────┘   └────────────────┘   └─────────┘
                                                                          │       │
                                                              file_summaries   chunks
                                                              collection       collection
                                                              (routing)        (retrieval)

QUERY  (a chat message, POST /chats/{id}/messages)
question
  → CLASSIFY       one LLM call: meta/broad/live-data/report/chart flags, decompose or
                    rewrite standalone against history
  → CACHE           exact + semantic match against this space's prior turn-1 answers
  → live-data/report? → fetch Redis/Postgres directly, answer from real data, skip below
  → ROUTE           embed question, search file_summaries → top-8 file paths
  → HYBRID SEARCH   dense cosine + BM25 fused (4:1 weighting), scoped to those files
  → CROSS-RERANK    CrossEncoder scores (question, chunk) pairs jointly, MMR-diversified
  → COMPRESS        one batched LLM call picks relevant line numbers for every chunk
  → GENERATE        LLM answers only from the compressed chunks, cites [file:Lstart-Lend]
  → ATTRIBUTE       claim-level citation check, computed after generation
  → return          streamed text + citations[] + table/chart + faithfulness trace
```

**Module map** (mirrors the flow above; grouped by directory, not exhaustive):

```
src/
├── config.py                  pydantic Settings — every knob in one place
├── auth.py                    password hashing, JWT sessions, space-access checks
├── crypto.py                  Fernet encryption for connector passwords at rest
├── llm_client.py               any OpenAI-compatible /chat/completions endpoint
├── pipeline.py                  wires every stage into ingest / query / live-data / report
├── trace.py                    dataclasses for every pipeline stage's trace output
├── cli.py                      `ingest` CLI, for local dev without the web UI
│
├── api/                         FastAPI routers
│   ├── main.py                    app wiring, CORS, model preload
│   ├── auth_routes.py              signup/login/logout/me
│   ├── user_routes.py              admin user list, role/space assignment, per-user insights
│   ├── routes.py                   spaces, sources (upload/ingest/trace)
│   ├── chat_routes.py               chats, messages (SSE), message trace/vectors
│   ├── connector_routes.py          Postgres/Redis connector CRUD + credential test
│   └── insights_routes.py           per-space / per-chat insights aggregation
│
├── connectors/
│   ├── live_data.py               place/device doc parsing, Redis fetch, live-data prompt
│   └── reports.py                 time-window resolution, Postgres fetch, report tool
│
├── ingest/
│   ├── repo_loader.py             clone + walk, skips node_modules/vendor/lockfiles/binaries
│   ├── ast_chunker.py             tree-sitter split on function/class boundaries
│   ├── chunker.py                 recursive character-split fallback, and PDF/DOCX/CSV parsing
│   ├── summarizer.py              per-file + per-repo LLM summaries (feeds routing)
│   └── contextualizer.py          one-sentence "where this fits" header per chunk
│
├── index/
│   ├── schema.py                   CodeChunk, FileSummary
│   ├── embedder.py                  sentence-transformers wrapper (lazy-loaded)
│   ├── vector_store.py              thin Qdrant wrapper
│   ├── doc_index.py                 file_summaries collection (routing layer)
│   └── chunk_index.py               chunks collection (retrieval layer)
│
├── retrieve/
│   ├── router.py                   question → shortlisted file paths
│   ├── hybrid_search.py             dense + BM25 fusion, file-scoped
│   ├── cross_reranker.py            CrossEncoder top-k → MMR-diversified final selection
│   └── compressor.py                line-level trim, all chunks in one call
│
└── generate/
    ├── decomposer.py                turn-1 classification: meta/broad/live/report/chart, split
    ├── rewriter.py                  turn-2+ classification + history-aware standalone rewrite
    ├── answer.py                    prompt build, table/chart parsing
    ├── provenance.py                ClaimAttributor — per-claim citation + verification
    └── faithfulness.py              whole-answer supported/unsupported check (eval-only)
```

---

## What makes it better than "stuff it in a vector DB"

- **Code-shaped chunks, not character-shaped ones.** `ast_chunker.py` walks
  the tree-sitter AST and cuts on `function_definition`/`class_definition`
  (and their per-language equivalents), so a chunk is always a whole
  function or class, never half of one. Oversized chunks are split on
  logical sub-blocks with line numbers preserved, not re-numbered from zero.
- **Two-stage retrieval, not one big flat search.** The router narrows a
  question to ~8 files using file-level summaries *before* any chunk-level
  search happens — chunk search then only competes against chunks from
  those files, which is what keeps unrelated-file noise out of the top
  candidates.
- **Hybrid, not pure embeddings.** Dense cosine similarity misses exact
  identifier matches (`verify_token`, `QDRANT_URL`); BM25 catches those.
  They're fused with a 4:1 dense:BM25 weighting rather than picking one.
- **A second, more expensive pass on a small candidate set.** Cross-encoder
  reranking (joint question+chunk attention) is too slow to run on a whole
  space, but cheap on a dozen candidates — so it only runs after routing +
  hybrid search have already done the culling.
- **Compression preserves citation accuracy.** The compressor asks the LLM
  for *line numbers*, not rewritten text, and narrows to the min..max window
  of relevant lines. That means `start_line`/`end_line` stay real line
  numbers all the way to the final citation — never resummarized text.
- **Line-accurate citations, always.** Every transform (chunking → context
  header → compression) carries `start_line`/`end_line` through unchanged,
  so a citation like `[src/retrieve/router.py:L14-L17]` points at exactly
  those lines in the source file.
- **Live/historical answers are built from real data, not narrated blind.**
  A live-data table's numbers come straight from Redis; a historical
  report's come straight from Postgres aggregate tables, computed in code
  (`build_report_block`) rather than asked of the LLM — removing the two
  failure modes that showed up when generation was trusted to format them:
  malformed JSON on a long multi-series table, and the model dropping the
  required block despite real data existing.
- **Faithfulness checking covers what the answer actually said.** A
  live-data table's numeric readings are checked exactly like prose claims
  are — a supported-vs-fabricated distinction the naive "only check the
  text field" version misses entirely, since the anti-duplication prompt
  rule pushes most of a live-data answer's real content into the table.
  Placeholder cells ("unavailable" for a metric that only exists on an
  inlet, never an outlet) are excluded rather than flagged unsupported —
  an absence isn't a claim needing grounding.
- **Local embeddings, hosted generation.** Embedding and reranking run on-device
  via `sentence-transformers`; only generation leaves the machine. Every model is
  a `.env` value — no code changes to swap one.
- **Two-tier generation.** Ingest writes one summary per file *and* per chunk,
  all of them one-sentence rewrites rather than reasoning — so `LLM_BULK_MODEL` can
  be the cheapest flash model while `LLM_MODEL` handles compression, answering,
  and question classification.
- **Retrieval can say "no".** Every stage before reranking returns top-k
  unconditionally, so an off-topic question would otherwise retrieve several chunks
  of arbitrary content. If the best reranked chunk scores below
  `RERANK_MIN_TOP_SCORE`, the query short-circuits without calling the LLM at
  all. The gate is on the *top* score, not each chunk's, because supporting context
  legitimately scores low and is still worth sending.
- **A refresh cancels the work.** Threads can't be killed, so the pipeline
  checks for a disconnected client before each LLM call and during retry
  backoffs. An abandoned query stops at its next checkpoint instead of holding
  rate-limit budget the next one needs.
- **Degrades instead of failing.** Every LLM stage has a fallback, so the pipeline
  still ingests and retrieves with no generation backend at all — you get template
  summaries and uncompressed chunks, but a real, inspectable prompt.

---

## Quickstart

Requires only Docker and an API key for any OpenAI-compatible provider — a free
[Google AI Studio key](https://aistudio.google.com/apikey) needs no credit card.

### Everything in Docker (backend + frontend + Qdrant)

```bash
git clone <this-repo> && cd GitHub-repo-Q-A-assistant
# .env needs LLM_API_KEY and JWT_SECRET — see Configuration above
echo 'LLM_API_KEY=your-key-here' >> .env
echo "JWT_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" >> .env
docker compose up --build     # or: make up
```

Open <http://localhost:5183> and sign up — **the first account created
becomes admin automatically**, and there's no seed/default account on a
fresh database. A regular signup after that gets no space access until an
admin assigns one from the Users page.

### Choosing a provider

Generation goes through any endpoint that serves OpenAI-style
`/chat/completions`, so switching provider is four lines in `.env` — no code
change:

| Provider | `LLM_BASE_URL` |
|---|---|
| Gemini (free tier) | `https://generativelanguage.googleapis.com/v1beta/openai` |
| OpenAI | `https://api.openai.com/v1` |
| Groq (free tier) | `https://api.groq.com/openai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Ollama (local) | `http://localhost:11434/v1` — leave `LLM_API_KEY` blank |

`LLM_MODEL` answers questions, compresses chunks, and classifies/decomposes
questions; `LLM_BULK_MODEL` writes the per-file and per-chunk summaries at
ingest plus claim-attribution checks. The second runs hundreds of times, so
point it at the cheapest model that can write a sentence — they need not be
the same provider tier, or even the same provider if you run two instances.

That is the whole setup — no Python, Node, or Qdrant install. Compose builds both
images, waits for Qdrant's healthcheck, and starts the backend and UI. The
`LLM_*`, `JWT_SECRET`, `CONNECTOR_ENCRYPTION_KEY`, and `QDRANT_URL` settings reach
the containers by interpolation; an exported shell variable of the same name
overrides `.env`. `.env` is gitignored — keep it that way.

Gemini's free tier is rate limited (~15 req/min, ~1500/day, **per model**). An ingest
makes one call per file *and* per chunk, so expect throttling on large sources; the
client retries per-minute 429s with backoff and fails fast on the daily quota,
which no amount of retrying will clear. Every LLM stage falls back rather than
failing, so the pipeline still ingests and retrieves with no key at all — you get
template summaries and uncompressed chunks, but a real, inspectable prompt.

UI on <http://localhost:5183>, API on <http://localhost:8000>. The first build
takes a few minutes (CPU torch wheel); the first *start* then spends ~15s loading
the embedding and reranker models, so the first question isn't charged for it.
Later runs reuse the cached layers and the `hf_cache` volume.

```bash
docker compose logs -f backend   # stage-by-stage retrieval logs
docker compose down              # stop, keep ingested data
docker compose down -v           # stop and wipe models, vectors, and ingests
```

All state lives in named volumes (`qdrant_storage`, `repo_data`, `uploads`, `app_db`,
`hf_cache`), so nothing is written into the working tree and `down -v` is the
single reset switch. Wiping only *some* of them leaves the SQLite metadata and Qdrant
vectors disagreeing; the API returns a 409 telling you to re-ingest that source.

### Local Python, Qdrant in Docker

For development, where you want `--reload` and the test suite. Requires Python
3.11+ and Node 20+.

```bash
docker compose up -d qdrant      # just the vector store
pip install -e ".[dev]"
echo 'LLM_API_KEY=your-key-here' >> .env
echo "JWT_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" >> .env

make serve                       # API on :8000
cd frontend && npm install && npm run dev   # UI on :5183
```

Set `PRELOAD_MODELS=false` while developing so `--reload` doesn't spend ~15s
loading models on every code edit. Sign up through the UI to create the admin
account (or `make ingest REPO=<git-url>` for a bare CLI ingest without the
web app — it writes into the same Qdrant/SQLite state, but has no space/auth
concept of its own).

Run the test suite:

```bash
make test
```

---

## UI

A React frontend (`frontend/`) with two audiences:

- **The app itself** — sign up/log in, browse spaces, add sources (clone a
  repo, upload PDF/DOCX/CSV, paste text, or configure a Redis/Postgres
  connector), and chat with streamed answers, citations, tables, and charts.
  Admins get a Users page (roles, space assignment) and an Insights
  dashboard (per-space and per-user retrieval/cache/latency/token/
  faithfulness metrics over a date range, drilling down to individual chats
  and questions).
- **The pipeline visualizer** — reached from a question's trace — walks
  through retrieval stage by stage (embed → route → hybrid search → rerank
  → compress → final prompt → claim-level citations, or the live-data/report
  tool call in place of retrieval) with a rotatable 3D PCA projection of the
  embedding space at each step. It's read from the same trace already
  computed for the real answer, not a separate no-generation dry run.

```bash
# with the backend already running (make serve)
cd frontend
npm install
npm run dev   # http://localhost:5183
```

Or via `docker compose up -d`, which starts Qdrant, the backend, and the
frontend together (frontend on port 5183).

---

## Design decisions & trade-offs

- **Local-first stack.** The pipeline is written against `Embedder` /
  `LLMClient` seams specifically so the local defaults (`bge-base-en-v1.5`,
  `bge-reranker-base`) can be swapped for hosted equivalents (Voyage
  `voyage-code-3`, Anthropic Claude) via `.env` only — no code changes.
- **No ColBERT / late-interaction stage.** RAGatouille adds real precision on
  code identifiers but also a heavy dependency and a Linux-only runtime
  constraint. Hybrid (dense + BM25) + cross-encoder reranking gets most of
  the benefit with a much smaller footprint; ColBERT is the natural next
  retrieval stage to bolt on if hybrid + rerank isn't precise enough.
- **Compression narrows to a contiguous line window**, not a splice of
  scattered relevant lines — simpler, and avoids handing the generator
  Frankenstein'd code with gaps that could be misread as contiguous.
- **`--depth 1` clone, size-filtered walk.** No git history, no generated
  files, vendored deps, or lockfiles enter the index — keeps ingestion time
  and noise down on large repos.
- **Live/report answers are built in code, not asked of the LLM.** Once the
  right Redis keys or Postgres rows are fetched, formatting them into a
  table/chart is deterministic string/JSON work — an LLM call to do the same
  thing only adds a chance of malformed output or a dropped block, confirmed
  live and repeatedly during development.
- **JWT over server-side sessions.** No session table to manage or clean up;
  a `token_version` column on `users` makes a token revocable (role change,
  space unassignment) without needing a blocklist — the next request with a
  stale token is rejected by comparing versions, not by expiry alone.
- **Faithfulness checking runs after generation, not as inline citations.**
  An earlier inline-citation-in-the-answer-prompt approach was dropped: a
  separate pass (`ClaimAttributor`) that re-reads the finished answer against
  the source is more reliable than asking the same call that's busy writing
  prose to also self-report its own sourcing.

---

## Limitations & future work

- **No incremental re-index.** Re-adding a source re-processes it in full;
  there's no diff-based update path yet.
- **No evaluation harness for the newer tool-calling paths.** `make eval`
  (`evals/runner.py` + `evals/golden_set.py`) scores the core retrieval
  pipeline against a golden set; live-data and historical-report answers
  aren't covered by it yet.
- **Single SQLite file, single Qdrant instance.** Fine at the scale this is
  built for; a multi-instance deployment would need both swapped for
  networked equivalents (Postgres, a clustered Qdrant) — the code has no
  built-in sharding.
- **Connector credentials are per-space, not centrally rotated.** Changing a
  database password means re-entering it in every space's connector, one at
  a time.
