from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App-wide configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Any OpenAI-compatible /chat/completions endpoint — see README for base URLs.
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_api_key: str = ""
    llm_model: str = "gemini-flash-latest"
    # Ingest runs one call per file *and* per chunk, so this one should be the cheapest
    # model that can write a sentence.
    llm_bulk_model: str = "gemini-flash-lite-latest"

    # ~18s of weight loading; at startup rather than charged to the first question.
    # Turn off in development so `--reload` doesn't pay it on every edit.
    preload_models: bool = True

    repo_clone_dir: str = "./data/repos"
    upload_dir: str = "./data/uploads"
    db_path: str = "./data/app.db"
    chunk_max_chars: int = 2000
    chunk_overlap: int = 200
    # How many prior turns (user+assistant pairs) are sent to the LLM as chat history.
    history_turns: int = 3

    # Hierarchical routing: how many files the router shortlists per question.
    top_files: int = 8
    # Hybrid retrieval: candidate pool size before reranking, and BM25:dense fusion weight
    # (4:1 dense:bm25 mirrors Anthropic's contextual-retrieval cookbook finding).
    # Reranking costs ~1s/candidate on CPU; the final top_k has consistently come from
    # fused positions <=16, so reranking all 40 was ~25s of pure waste per query.
    hybrid_candidate_k: int = 12
    hybrid_dense_weight: float = 1.0
    hybrid_bm25_weight: float = 0.25
    # Cross-encoder reranking: final number of chunks handed to compression/generation.
    rerank_top_k: int = 6
    # Compound questions ("what does X do, and how is Y different?") are split into at
    # most this many independent sub-questions, each retrieved in parallel, before the
    # shared compress/generate steps. 1 disables decomposition entirely.
    max_subquestions: int = 3
    # Semantic cache: on an exact-match miss, a turn-1 question is compared (cosine) to
    # every previously-cached turn-1 question in the space; a hit above this score reuses
    # that answer with zero retrieval/generation. Deliberately conservative — a false
    # positive here serves a confidently WRONG answer, unlike a miss which only costs
    # time. Tune down only after checking real near-miss pairs, not by feel.
    semantic_cache_min_score: float = 0.92
    # If the best reranked chunk scores below this, nothing in the repo answers the
    # question. Measured: off-topic tops out at 0.0, weakest real match was 0.35.
    rerank_min_top_score: float = 0.01
    # Disambiguates zero reranked chunks: broad question (go wide) vs. off-topic (say
    # NO_MATCH). Below this route score, treat it as off-topic. Measured across 3 spaces,
    # 12 questions (evals/golden_set.py): off-topic tops out at 0.4876, on-topic (broad
    # and factual) bottoms out at 0.5351 — still a real gap, just off-center from the
    # first 0.48 guess. Re-check with `make eval` if this space's content shifts.
    route_min_top_score: float = 0.50
    # Whole-document fallback (e.g. "summarize the story"): no single chunk answers such a
    # question, so the rerank gate above always rejects it. When that happens, the routed
    # file(s) are sent whole instead of chunk-by-chunk — but only up to this token budget,
    # kept well under typical per-minute token quotas rather than the model's raw context
    # window, which is usually the looser constraint on a free tier.
    wide_answer_max_tokens: int = 150_000


settings = Settings()
