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
    # Cached ingest traces, keyed by repo name — lets a previously-ingested repo be
    # reselected in the UI without re-cloning/re-summarizing/re-embedding.
    ingest_cache_dir: str = "./data/ingest_cache"
    chunk_max_chars: int = 2000
    chunk_overlap: int = 200

    # Hierarchical routing: how many files the router shortlists per question.
    top_files: int = 8
    # Hybrid retrieval: candidate pool size before reranking, and BM25:dense fusion weight
    # (4:1 dense:bm25 mirrors Anthropic's contextual-retrieval cookbook finding).
    hybrid_candidate_k: int = 40
    hybrid_dense_weight: float = 1.0
    hybrid_bm25_weight: float = 0.25
    # Cross-encoder reranking: final number of chunks handed to compression/generation.
    rerank_top_k: int = 6
    # If the best reranked chunk scores below this, nothing in the repo answers the
    # question. Measured: off-topic tops out at 0.0, weakest real match was 0.35.
    rerank_min_top_score: float = 0.01


settings = Settings()
