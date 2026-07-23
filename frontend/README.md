# RAG Pipeline Visualizer

React + Vite frontend for the [GitHub Repo Q&A Assistant](../README.md). It
doesn't chat with the LLM — it traces the retrieval pipeline itself, stage by
stage, so you can see what the assistant sees:

- **Ingestion tab** — clone → walk files → AST-chunk → summarize/contextualize
  → embed + upsert, with the chunks and embeddings produced at each step.
- **Retrieval tab** — embed question → route to files → hybrid search →
  cross-rerank → compress → final prompt, ending at the exact prompt that
  would be sent to the LLM (generation itself is never called).
- A fullscreen vector-space view (PCA-projected to 2D) showing where the
  question and candidate chunks land relative to each other at each stage.

## Running

Requires the backend running first (see the [root README](../README.md)).

```bash
npm install
npm run dev      # http://localhost:5183
```

The API base URL is hardcoded to `http://localhost:8000` in `src/api.js`.

## Layout

```
src/
├── api.js                       fetch wrappers: /repos, /ingest/trace(/stream), /query/trace
├── App.jsx                       tab state, stage runner wiring, layout
├── hooks/useStageRunner.js       drives the step-by-step reveal/highlight animation
├── views/
│   ├── IngestView.jsx             ingestion stage list + sections
│   └── RetrievalView.jsx          retrieval stage list + sections
└── components/
    ├── RagAtoms.jsx                small shared UI pieces (cards, tags, score pills)
    ├── RagVectorSpace.jsx          2D embedding scatter plot
    └── RagVectorModal.jsx          fullscreen vector-space walkthrough
```

## Scripts

```bash
npm run dev       # dev server with HMR
npm run build     # production build
npm run lint      # oxlint
npm run preview   # preview a production build
```
