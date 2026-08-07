# GitHub Repo Q&A Assistant

Point it at a public GitHub repo, ask questions in plain English, get answers
with citations back to the exact file and line range — no hallucinated APIs,
no "I looked at the whole repo and guessed."

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
INGEST  (make ingest REPO=<git-url>, run once per repo)
┌──────────┐   ┌────────────┐   ┌─────────────┐   ┌────────────────┐   ┌─────────┐
│ git clone│ → │ walk files │ → │ AST-chunk   │ → │ summarize file  │ → │ embed + │
│ (depth 1)│   │ (ext/size  │   │ (tree-sitter│   │ + repo (LLM),   │   │ upsert  │
│          │   │  filtered) │   │  per lang)  │   │ contextualize   │   │ to      │
│          │   │            │   │             │   │ each chunk (LLM)│   │ Qdrant  │
└──────────┘   └────────────┘   └─────────────┘   └────────────────┘   └─────────┘
                                                                          │       │
                                                              file_summaries   chunks
                                                              collection       collection
                                                              (routing)        (retrieval)

QUERY  (make serve, POST /query)
question
  → ROUTE          embed question, search file_summaries → top-8 file paths
  → HYBRID SEARCH   dense cosine + BM25 fused (4:1 weighting), scoped to those files → top-40
  → CROSS-RERANK    CrossEncoder scores (question, chunk) pairs jointly → top-6
  → COMPRESS        one batched LLM call picks relevant line numbers for every chunk
  → GENERATE        LLM answers only from the compressed chunks, cites [file:Lstart-Lend]
  → return          { text, citations[], confidence }
```

**Module map** (mirrors the flow above):

```
src/
├── config.py                 pydantic Settings — every knob in one place
├── llm_client.py              any OpenAI-compatible /chat/completions endpoint
├── pipeline.py                 wires every stage into ingest_repo() / query()
├── cli.py, api/                CLI (`ingest`, `prompt`) and FastAPI (`/ingest`, `/query`, `/health`)
├── cancellation.py             stops in-flight work when the client disconnects
│
├── ingest/
│   ├── repo_loader.py          clone + walk, skips node_modules/vendor/lockfiles/binaries
│   ├── ast_chunker.py          tree-sitter split on function/class boundaries
│   ├── chunker.py              recursive character-split fallback for unsupported languages
│   ├── summarizer.py           per-file + per-repo LLM summaries (feeds routing)
│   └── contextualizer.py       one-sentence "where this fits" header per chunk
│
├── index/
│   ├── schema.py                CodeChunk, FileSummary
│   ├── embedder.py               sentence-transformers wrapper (lazy-loaded)
│   ├── vector_store.py           thin Qdrant wrapper
│   ├── doc_index.py              file_summaries collection (routing layer)
│   └── chunk_index.py            chunks collection (retrieval layer)
│
├── retrieve/
│   ├── router.py                 question → shortlisted file paths
│   ├── hybrid_search.py          dense + BM25 fusion, file-scoped
│   ├── cross_reranker.py         CrossEncoder top-40 → top-6, gated on the best score
│   └── compressor.py             line-level trim, all chunks in one call
│
└── generate/
    └── answer.py                 prompt build, citation parsing, confidence
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
  those files, which is what keeps unrelated-file noise out of the top-40.
- **Hybrid, not pure embeddings.** Dense cosine similarity misses exact
  identifier matches (`verify_token`, `QDRANT_URL`); BM25 catches those.
  They're fused with a 4:1 dense:BM25 weighting rather than picking one.
- **A second, more expensive pass on a small candidate set.** Cross-encoder
  reranking (joint question+chunk attention) is too slow to run on a whole
  repo, but cheap on 40 candidates — so it only runs after routing + hybrid
  search have already done the culling.
- **Compression preserves citation accuracy.** The compressor asks the LLM
  for *line numbers*, not rewritten text, and narrows to the min..max window
  of relevant lines. That means `start_line`/`end_line` stay real line
  numbers all the way to the final citation — never resummarized text.
- **Line-accurate citations, always.** Every transform (chunking → context
  header → compression) carries `start_line`/`end_line` through unchanged,
  so a citation like `[src/retrieve/router.py:L14-L17]` points at exactly
  those lines in the source file.
- **Local embeddings, hosted generation.** Embedding and reranking run on-device
  via `sentence-transformers`; only generation leaves the machine. Every model is
  a `.env` value — no code changes to swap one.
- **Two-tier generation.** Ingest writes one summary per file *and* per chunk
  (224 calls on this repo, ~1400 on a 200-file one), all of them one-sentence
  rewrites rather than reasoning — so `LLM_BULK_MODEL` can be the cheapest
  flash model while `LLM_MODEL` handles compression and answering. A query
  costs two calls: one batched compression, one answer.
- **Retrieval can say "no".** Every stage before reranking returns top-k
  unconditionally, so an off-topic question would otherwise retrieve six chunks
  of arbitrary code. If the best reranked chunk scores below
  `RERANK_MIN_TOP_SCORE`, the query short-circuits without calling the LLM at
  all. The gate is on the *top* score, not each chunk's — supporting context
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
# create .env — see Configuration below; the only required line is:
echo 'LLM_API_KEY=your-key-here' > .env
docker compose up --build     # or: make up
```

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

`LLM_MODEL` answers questions and compresses chunks; `LLM_BULK_MODEL` writes the
per-file and per-chunk summaries at ingest. The second runs hundreds of times, so
point it at the cheapest model that can write a sentence — they need not be the
same provider tier, or even the same provider if you run two instances.

That is the whole setup — no Python, Node, or Qdrant install. Compose builds both
images, waits for Qdrant's healthcheck, and starts the backend and UI. The
`LLM_*` and `QDRANT_URL` settings reach the containers by interpolation; an
exported shell variable of the same name overrides `.env`. `.env` is gitignored —
keep it that way.

Gemini's free tier is rate limited (~15 req/min, ~1500/day, **per model**). An ingest
makes one call per file *and* per chunk, so expect throttling on large repos; the
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

All state lives in named volumes (`qdrant_storage`, `repo_data`, `ingest_cache`,
`hf_cache`), so nothing is written into the working tree and `down -v` is the
single reset switch. Wiping only *some* of them leaves the trace cache and vector
store disagreeing; the API returns a 409 telling you to re-ingest.

### Local Python, Qdrant in Docker

For development, where you want `--reload` and the test suite. Requires Python
3.11+ and Node 20+.

```bash
docker compose up -d qdrant      # just the vector store
pip install -e ".[dev]"
echo 'LLM_API_KEY=your-key-here' > .env

make ingest REPO=https://github.com/psf/requests
make serve                       # API on :8000

cd frontend && npm install && npm run dev   # UI on :5183
```

Set `PRELOAD_MODELS=false` while developing so `--reload` doesn't spend ~15s
loading models on every code edit.

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How is a redirect followed?", "repo": "requests"}'
```

Run the test suite:

```bash
make test
```

---

## Configuration

Everything lives in a `.env` file in the repo root (gitignored — never commit it).
Every setting has a working default, so **`LLM_API_KEY` is the only line you must
write**. Copy the block below and change what you need:

```bash
# ── Generation ────────────────────────────────────────────────────────────────
LLM_API_KEY=your-key-here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-flash-latest
LLM_BULK_MODEL=gemini-flash-lite-latest

# ── Local models (downloaded on first run, cached after) ──────────────────────
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
PRELOAD_MODELS=true

# ── Storage ───────────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
REPO_CLONE_DIR=./data/repos
INGEST_CACHE_DIR=./data/ingest_cache

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_MAX_CHARS=2000
CHUNK_OVERLAP=200

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_FILES=8
HYBRID_CANDIDATE_K=40
HYBRID_DENSE_WEIGHT=1.0
HYBRID_BM25_WEIGHT=0.25
RERANK_TOP_K=6
RERANK_MIN_TOP_SCORE=0.01
```

### Generation

| Setting | Default | What it does |
|---|---|---|
| `LLM_API_KEY` | *(none)* | **Required.** Leave blank only for Ollama, which needs no auth. |
| `LLM_BASE_URL` | Gemini | Any OpenAI-compatible `/chat/completions` host — see the table above. |
| `LLM_MODEL` | `gemini-flash-latest` | Answers questions and compresses chunks. ~2 calls per question. |
| `LLM_BULK_MODEL` | `gemini-flash-lite-latest` | Writes the per-file and per-chunk summaries at ingest. Runs **hundreds of times** (224 calls for this repo), so use the cheapest model that can write a sentence. |

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
| `INGEST_CACHE_DIR` | `./data/ingest_cache` | Saved ingest traces, so re-selecting a repo in the UI replays instantly instead of re-ingesting. |

### Chunking

| Setting | Default | What it does |
|---|---|---|
| `CHUNK_MAX_CHARS` | `2000` | Ceiling before a chunk is split further. Bigger chunks mean more context per hit but a longer final prompt. |
| `CHUNK_OVERLAP` | `200` | Characters repeated between split chunks, so a definition cut across a boundary still appears whole in one of them. |

### Retrieval

| Setting | Default | What it does |
|---|---|---|
| `TOP_FILES` | `8` | Files the router shortlists before any chunk search runs. |
| `HYBRID_CANDIDATE_K` | `40` | Candidates passed to reranking. **This sets query latency** — reranking costs ~1s per candidate on CPU. |
| `HYBRID_DENSE_WEIGHT` | `1.0` | Weight on cosine similarity in the fusion. |
| `HYBRID_BM25_WEIGHT` | `0.25` | Weight on keyword overlap. The 4:1 ratio mirrors Anthropic's contextual-retrieval finding. |
| `RERANK_TOP_K` | `6` | Chunks that reach compression and the final prompt. |
| `RERANK_MIN_TOP_SCORE` | `0.01` | If the *best* reranked chunk scores below this, nothing in the repo answers the question and the query short-circuits without calling the LLM. Gates the top score rather than each chunk's, because supporting context legitimately scores low. |

---

## UI — RAG Pipeline Visualizer

A React frontend (`frontend/`) that walks through ingestion and retrieval
stage by stage — clone → walk → chunk → contextualize → embed on one tab,
embed question → route → hybrid search → rerank → compress → final prompt →
answer on the other — with a rotatable 3D PCA projection of the embedding space
at each step. It
calls trace endpoints (`/ingest/trace(/stream)`, `/query/trace`, `/repos`)
that mirror the real pipeline but stop before the LLM generation call, so you
see exactly what would be sent without spending a generation.

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

---

## Limitations & future work

- **Single repo per session.** No cross-repo queries — one `repo` tag per
  Qdrant collection, scoped at ingest and query time.
- **No incremental re-index.** Re-running `ingest` re-clones and
  re-processes every file; there's no diff-based update path yet.
- **No evaluation harness yet.** `PROJECT_SPEC.md` §7 specs a golden-set +
  `ragas` comparison against a naive baseline (fixed-size chunking, no
  routing/rerank/compression) — that's the natural next milestone to
  quantify what each stage is actually worth.
- **Visualizer, not a chat UI.** The frontend traces and displays every
  pipeline stage up through the final assembled prompt, but stops short of
  calling the LLM — there's no chat window that shows a generated, cited
  answer yet.
