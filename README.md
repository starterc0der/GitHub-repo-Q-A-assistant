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

100% local: Ollama for generation, `sentence-transformers` for embeddings and
reranking, Qdrant for vector storage. No API keys required.

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
  → COMPRESS        LLM picks question-relevant line numbers per chunk, drops the rest
  → GENERATE        LLM answers only from the compressed chunks, cites [file:Lstart-Lend]
  → return          { text, citations[], confidence }
```

**Module map** (mirrors the flow above):

```
src/
├── config.py                 pydantic Settings — every knob in one place
├── llm_client.py              thin wrapper over Ollama's HTTP API
├── pipeline.py                 wires every stage into ingest_repo() / query()
├── cli.py, api/                CLI (`ingest`) and FastAPI (`/ingest`, `/query`, `/health`)
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
│   ├── cross_reranker.py         CrossEncoder top-40 → top-6
│   └── compressor.py             line-level trim to question-relevant code
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
- **Fully local, zero API keys.** Generation runs through Ollama, embeddings
  and reranking run through `sentence-transformers`, and the only external
  network call in the query path is to your own local Qdrant. Swap
  `EMBEDDING_MODEL` / `OLLAMA_MODEL` in `.env` to try different local models
  with no code changes.

---

## Quickstart

Requires Docker, [Ollama](https://ollama.com), and Python 3.11+.

```bash
# 1. start Qdrant
docker compose up -d

# 2. pull a local model for generation
ollama pull llama3.1

# 3. install deps
pip install -e ".[dev]"
cp .env.example .env

# 4. ingest a repo
make ingest REPO=https://github.com/psf/requests

# 5. serve the API
make serve
```

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

## Design decisions & trade-offs

- **Local-first stack.** The pipeline is written against `Embedder` /
  `LLMClient` seams specifically so the local defaults (`bge-base-en-v1.5`,
  `bge-reranker-base`, Ollama) can be swapped for hosted equivalents (Voyage
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
- **No UI.** Currently CLI (`ingest`) + FastAPI (`/query`) only; a chat
  front-end with a sources panel is unbuilt.
