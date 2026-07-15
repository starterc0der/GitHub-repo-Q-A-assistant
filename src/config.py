from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App-wide configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    repo_clone_dir: str = "./data/repos"
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


settings = Settings()
