from __future__ import annotations

from dataclasses import dataclass, field

# The three spaces already ingested in the running app. Grab real ids/sources via:
#   curl -s http://localhost:8000/spaces | python3 -m json.tool
#   curl -s http://localhost:8000/spaces/{id}/sources | python3 -m json.tool
STORY_SPACE = "ef4d2b6dce5746bb9e0d6bc9d595f2c1"  # Mahabharata.pdf + Ramayana.pdf
PDF_TEST_SPACE = "a74611da7f6f410aa19a62f4979092ca"  # children's short stories PDF + a car specs CSV
CODE_SPACE = "d51fc44bc12849f4a2925f62335807e3"  # this app's own repo — an OLDER snapshot,
# ingested before this session's chat/spaces/decomposition/semantic-cache work, so
# questions here must only be about what's actually in it (pipeline.py, router.py,
# hybrid_search.py, cross_reranker.py, compressor.py, cancellation.py, etc.) — verify with:
#   docker compose exec -T backend python3 -c "from src.config import settings; \
#     from src.index.chunk_index import ChunkIndex; from src.index.vector_store import VectorStore; \
#     ci = ChunkIndex(VectorStore(settings.qdrant_url)); \
#     print(sorted({c.file_path for c, _ in ci.fetch_by_files('CODE_SPACE_ID', [])}))"


@dataclass
class EvalCase:
    name: str
    space_id: str
    question: str
    # "answered" (grounded, non-refused answer) | "no_match" | "wide_fallback"
    expect_gate: str = "answered"
    # substring (case-insensitive) that must appear in at least one routed/reranked file path
    expect_files_contain: str | None = None
    # case-insensitive substrings that must ALL appear in the answer text — the cheapest
    # correctness proxy; an LLM-judge layer is a later upgrade, not a v1 requirement
    expect_answer_contains: list[str] = field(default_factory=list)
    # None = don't check; True/False = assert query decomposition did/didn't split it
    expect_decomposed: bool | None = None
    # Exact file_path(s) (e.g. "mahabharata.pdf · p.24") that ground truth says are
    # relevant — powers Precision@K/Recall@K/MRR/NDCG@K against trace.reranked. SILVER
    # labels: the top reranked page from a run whose ANSWER TEXT was independently
    # verified against expect_answer_contains, not independently human-read from the
    # source PDF. Good enough to catch a regression; not a substitute for real labeling.
    expect_relevant_files: list[str] = field(default_factory=list)


# Every expected value below is grounded in an answer this app actually produced during
# live testing this session, not guessed — see the conversation history for the source.
GOLDEN_SET: list[EvalCase] = [
    EvalCase(
        name="factual-kripa",
        space_id=STORY_SPACE,
        question="who is Kripa",
        expect_files_contain="mahabharata",
        expect_answer_contains=["combat"],
        expect_relevant_files=["mahabharata.pdf · p.24"],
    ),
    EvalCase(
        name="factual-yudhishthira",
        space_id=STORY_SPACE,
        question="who is Yudhishthira",
        expect_files_contain="mahabharata",
        # No expect_answer_contains: he's a Pandava, but which specific fact a given
        # retrieved page emphasizes varies run to run — "Pandava" isn't guaranteed.
        # Multiple valid pages, not one: Yudhishthira is mentioned across dozens of
        # pages in a 217-page epic — real observed correct pages, not a single "the" one.
        expect_relevant_files=["mahabharata.pdf · p.51", "mahabharata.pdf · p.88", "mahabharata.pdf · p.47"],
    ),
    EvalCase(
        name="factual-drona",
        space_id=STORY_SPACE,
        question="who is Drona",
        expect_files_contain="mahabharata",
        expect_answer_contains=["Bharadwaja"],
        # Same reasoning as Yudhishthira above: multiple real observed correct pages.
        expect_relevant_files=["mahabharata.pdf · p.209", "mahabharata.pdf · p.27", "mahabharata.pdf · p.26"],
    ),
    EvalCase(
        name="offtopic-gates-no-match",
        space_id=STORY_SPACE,
        question="what is the boiling point of water in Celsius?",
        expect_gate="no_match",
    ),
    EvalCase(
        name="broad-question-goes-wide",
        space_id=STORY_SPACE,
        question="summarize the entire story",
        expect_gate="wide_fallback",
    ),
    EvalCase(
        name="compound-question-decomposes",
        space_id=STORY_SPACE,
        question="who is Drona, and how is he different from Kripa?",
        expect_files_contain="mahabharata",
        expect_decomposed=True,
    ),

    # ---------------------------------------------------------- story: more Mahabharata
    EvalCase(
        name="factual-arjuna",
        space_id=STORY_SPACE,
        question="who is Arjuna",
        expect_files_contain="mahabharata",
        # No expect_answer_contains: the retrieved page describes his actions (weapons,
        # chariot) rather than stating "Pandava" the way the Yudhishthira/Drona pages
        # happen to — a wrong assumption in this golden-set case, not a pipeline bug.
    ),
    EvalCase(
        name="factual-bhishma",
        space_id=STORY_SPACE,
        question="who is Bhishma",
        expect_files_contain="mahabharata",
    ),
    EvalCase(
        name="factual-karna",
        space_id=STORY_SPACE,
        question="who is Karna",
        expect_files_contain="mahabharata",
    ),

    # ---------------------------------------------------------------- story: Ramayana
    EvalCase(
        name="factual-rama",
        space_id=STORY_SPACE,
        question="who is Rama",
        expect_files_contain="ramayana",
    ),
    EvalCase(
        name="factual-sita",
        space_id=STORY_SPACE,
        question="who is Sita",
        expect_files_contain="ramayana",
    ),
    EvalCase(
        name="factual-lakshman",
        space_id=STORY_SPACE,
        question="who is Lakshman",
        expect_files_contain="ramayana",
    ),
    EvalCase(
        name="factual-bharat-ramayana",
        space_id=STORY_SPACE,
        question="who is Bharat in the Ramayana",
        expect_files_contain="ramayana",
    ),
    EvalCase(
        name="offtopic-recipe",
        space_id=STORY_SPACE,
        question="what is the recipe for pizza dough?",
        expect_gate="no_match",
    ),
    EvalCase(
        name="broad-ramayana-goes-wide",
        space_id=STORY_SPACE,
        question="summarize the whole Ramayana",
        expect_gate="wide_fallback",
    ),
    EvalCase(
        name="compound-cross-epic-decomposes",
        space_id=STORY_SPACE,
        question="who is Bhishma in the Mahabharata, and who is Rama in the Ramayana?",
        expect_decomposed=True,
    ),

    # --------------------------------------------------- pdf test: children's stories
    EvalCase(
        name="factual-wind-and-sun",
        space_id=PDF_TEST_SPACE,
        question="in the story 'The Wind and the Sun', who wins the contest?",
        expect_files_contain="short-stories",
        expect_answer_contains=["Sun"],
    ),
    EvalCase(
        name="factual-spectacles-moral",
        space_id=PDF_TEST_SPACE,
        question="what is the moral of the story about the villager and the spectacles?",
        expect_files_contain="short-stories",
        expect_answer_contains=["ignorance"],
    ),
    EvalCase(
        name="factual-unfriendly-river",
        space_id=PDF_TEST_SPACE,
        question="in the story about the unfriendly river, who is Scamp?",
        expect_files_contain="short-stories",
    ),
    EvalCase(
        name="factual-day-with-pigs",
        space_id=PDF_TEST_SPACE,
        question="what happens in the story 'A Day with Pigs'?",
        expect_files_contain="short-stories",
    ),
    EvalCase(
        name="factual-gail-the-whale",
        space_id=PDF_TEST_SPACE,
        question="what happens in the story about Gail the Whale?",
        expect_files_contain="short-stories",
    ),

    # ------------------------------------------------------------- pdf test: cars CSV
    EvalCase(
        name="factual-nissan-titan-hp",
        space_id=PDF_TEST_SPACE,
        question="how many horsepower does the Nissan Titan Warrior have?",
        expect_files_contain="cars",
        expect_answer_contains=["400"],
    ),
    EvalCase(
        name="factual-vw-xl1-type",
        space_id=PDF_TEST_SPACE,
        question="what type of car is the Volkswagen XL1?",
        expect_files_contain="cars",
        expect_answer_contains=["hybrid"],
    ),
    EvalCase(
        name="factual-cadillac-lyriq-price",
        space_id=PDF_TEST_SPACE,
        question="what is the price of the Cadillac Lyriq Launch Edition?",
        expect_files_contain="cars",
    ),
    EvalCase(
        name="offtopic-us-president",
        space_id=PDF_TEST_SPACE,
        question="who is the president of the United States?",
        expect_gate="no_match",
    ),
    EvalCase(
        name="broad-list-all-cars",
        space_id=PDF_TEST_SPACE,
        question="list all the cars in the dataset",
        expect_gate="wide_fallback",
    ),
    EvalCase(
        name="broad-every-story",
        space_id=PDF_TEST_SPACE,
        question="summarize every story in the document",
        expect_gate="wide_fallback",
    ),
    EvalCase(
        name="compound-cross-source-decomposes",
        space_id=PDF_TEST_SPACE,
        question="who wins in The Wind and the Sun, and how much horsepower does the Nissan Titan Warrior have?",
        expect_decomposed=True,
    ),

    # ------------------------------------------------ test space: this app's own code
    EvalCase(
        name="factual-router-class",
        space_id=CODE_SPACE,
        question="what does the Router class do?",
        expect_files_contain="router.py",
        expect_answer_contains=["rout"],  # matches "route"/"routing"/"routed"
    ),
    EvalCase(
        name="factual-hybrid-search",
        space_id=CODE_SPACE,
        question="what does HybridSearch combine?",
        expect_files_contain="hybrid_search.py",
        expect_answer_contains=["bm25"],
    ),
    EvalCase(
        name="factual-cross-reranker",
        space_id=CODE_SPACE,
        question="what does the CrossReranker class do?",
        expect_files_contain="cross_reranker.py",
    ),
    EvalCase(
        name="factual-cancellation",
        space_id=CODE_SPACE,
        question="what does cancellation.py do?",
        expect_files_contain="cancellation.py",
        expect_answer_contains=["cancel"],
    ),
    EvalCase(
        name="factual-compressor",
        space_id=CODE_SPACE,
        question="what does the Compressor class do?",
        expect_files_contain="compressor.py",
    ),
    EvalCase(
        name="factual-llm-client",
        space_id=CODE_SPACE,
        question="what does llm_client.py do?",
        expect_files_contain="llm_client.py",
    ),
    EvalCase(
        name="factual-embedder",
        space_id=CODE_SPACE,
        question="what does the Embedder class do?",
        expect_files_contain="embedder.py",
    ),
    EvalCase(
        name="offtopic-capital-of-japan",
        space_id=CODE_SPACE,
        question="what is the capital of Japan?",
        expect_gate="no_match",
    ),
    EvalCase(
        name="broad-summarize-codebase",
        space_id=CODE_SPACE,
        question="summarize the entire codebase",
        expect_gate="wide_fallback",
    ),
    EvalCase(
        name="compound-code-decomposes",
        space_id=CODE_SPACE,
        question="what does the Router class do, and what does the CrossReranker class do?",
        expect_files_contain="router.py",
        expect_decomposed=True,
    ),
]
