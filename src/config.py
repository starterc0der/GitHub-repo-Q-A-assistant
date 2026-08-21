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
    # Encrypts connector passwords at rest (see src/crypto.py) — never plaintext in
    # sqlite. Generate with: python3 -c "from cryptography.fernet import Fernet;
    # print(Fernet.generate_key().decode())" and put it in .env. Empty by default so a
    # fresh checkout fails loudly on first connector use instead of silently storing
    # plaintext — there's no safe default to ship here.
    connector_encryption_key: str = ""
    # Signs login JWTs (see src/auth.py). Generate with: python3 -c "import secrets;
    # print(secrets.token_hex(32))" and put it in .env. Empty by default for the same
    # reason as connector_encryption_key — fail loudly on first login rather than sign
    # tokens with a guessable default.
    jwt_secret: str = ""
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
    # MMR relevance/diversity balance for that final selection (see CrossReranker) — 1.0
    # is plain top-k by score, 0.0 is pure diversity. 0.7 favors relevance but still lets
    # a near-duplicate chunk lose to a genuinely different, slightly lower-scoring one.
    mmr_lambda: float = 0.7
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
    # A sub-question's top score clearing rerank_min_top_score only proves relevance
    # (on-topic), not answerability (the specific fact is actually in there) — a chunk can
    # pass the floor and still not answer the question. Below this ceiling, worth a cheap
    # SufficiencyChecker call to actually read the evidence; at or above it, the score
    # alone is trusted (no measured calibration data yet, unlike rerank_min_top_score
    # above — this is a starting point, re-check with real near-miss cases via `make eval`
    # once there's a golden-set case for it).
    sufficiency_check_max_score: float = 0.4
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
    # Defensive ceiling on the normal (non-wide) answer path — rerank_top_k + compression
    # already keeps this well under budget in practice, but nothing enforced it. If ever
    # exceeded, lowest-ranked chunks are dropped first rather than silently over-filling
    # the prompt.
    answer_context_max_tokens: int = 12_000


settings = Settings()
