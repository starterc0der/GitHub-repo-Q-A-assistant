from __future__ import annotations

from src.config import Settings
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _CandidateState


def _chunk(chunk_id: str, file_path: str = "a.py") -> CodeChunk:
    return CodeChunk(
        id=chunk_id, space_id="demo", source_id="src1", file_path=file_path,
        language="text", symbol_name=None, start_line=1, end_line=1, code="x",
    )


def _pipeline_stub(rerank_top_k: int = 6) -> Pipeline:
    """_merge_candidates only touches settings.rerank_top_k, so this skips
    Pipeline.__init__ entirely rather than loading real embedding/reranker models."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings(rerank_top_k=rerank_top_k)
    return pipeline


def test_merge_candidates_passes_through_a_single_state_unchanged() -> None:
    pipeline = _pipeline_stub()
    state = _CandidateState(query_vector=[1.0], routed=[], scored_candidates=[], reranked_scored=[])

    merged, routed_origin, candidate_origin, reranked_origin = pipeline._merge_candidates(
        ["only question"], [state]
    )

    assert merged is state
    assert routed_origin == {} and candidate_origin == {} and reranked_origin == {}


def test_merge_candidates_unions_routed_files_keeping_the_max_score() -> None:
    pipeline = _pipeline_stub()
    state_a = _CandidateState(
        [0.0], routed=[("shared.py", "src1", 0.4), ("only_a.py", "src1", 0.9)],
        scored_candidates=[], reranked_scored=[],
    )
    state_b = _CandidateState([0.0], routed=[("shared.py", "src1", 0.7)], scored_candidates=[], reranked_scored=[])

    merged, routed_origin, _, _ = pipeline._merge_candidates(["q1", "q2"], [state_a, state_b])

    scores = {fp: score for fp, _sid, score in merged.routed}
    assert scores == {"shared.py": 0.7, "only_a.py": 0.9}
    assert routed_origin["shared.py"] == "q2"  # the higher-scoring sub-question wins
    assert routed_origin["only_a.py"] == "q1"


def test_merge_candidates_takes_the_slowest_branch_per_stage_not_the_sum() -> None:
    """Sub-question branches run concurrently (ThreadPoolExecutor), so a stage's real
    contribution to wall-clock time is whichever branch was slowest at that stage, not
    the total of all branches added together."""
    pipeline = _pipeline_stub()
    fast = _CandidateState(
        [0.0], routed=[], scored_candidates=[], reranked_scored=[],
        timings={"embed": 10.0, "route": 5.0, "hybrid": 20.0, "rerank": 100.0},
    )
    slow = _CandidateState(
        [0.0], routed=[], scored_candidates=[], reranked_scored=[],
        timings={"embed": 50.0, "route": 2.0, "hybrid": 15.0, "rerank": 90.0},
    )

    merged, _, _, _ = pipeline._merge_candidates(["q1", "q2"], [fast, slow])

    assert merged.timings == {"embed": 50.0, "route": 5.0, "hybrid": 20.0, "rerank": 100.0}


def test_merge_candidates_gives_every_sub_question_a_fair_share_of_reranked_slots() -> None:
    """Regression guard for the exact bug decomposition exists to fix: without a fair
    share, one sub-question's higher cross-encoder scores would crowd out a second
    sub-question's chunks entirely — even though cross-encoder scores from different
    queries were never comparable in the first place."""
    pipeline = _pipeline_stub(rerank_top_k=4)
    loud = _CandidateState([0.0], routed=[], scored_candidates=[], reranked_scored=[
        (_chunk("loud-1"), 0.99), (_chunk("loud-2"), 0.98),
        (_chunk("loud-3"), 0.97), (_chunk("loud-4"), 0.96),
    ])
    quiet = _CandidateState([0.0], routed=[], scored_candidates=[], reranked_scored=[
        (_chunk("quiet-1"), 0.10), (_chunk("quiet-2"), 0.09),
    ])

    merged, _, _, reranked_origin = pipeline._merge_candidates(["loud q", "quiet q"], [loud, quiet])

    ids = {chunk.id for chunk, _ in merged.reranked_scored}
    assert "quiet-1" in ids  # not crowded out despite the far lower absolute score
    assert reranked_origin["quiet-1"] == "quiet q"
    # share = ceil(4/2) = 2 per sub-question
    assert len([c for c in ids if c.startswith("loud")]) == 2
    assert len([c for c in ids if c.startswith("quiet")]) == 2
