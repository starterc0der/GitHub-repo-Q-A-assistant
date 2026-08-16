from __future__ import annotations

from src.index.schema import CodeChunk
from src.retrieve.cross_reranker import CrossReranker


def _chunk(file_path: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-1",
        space_id="demo",
        source_id="src1",
        file_path=file_path,
        language="python",
        symbol_name=None,
        start_line=1,
        end_line=1,
        code="def run(): pass",
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]):
        self.scores = scores

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        return self.scores


def test_rerank_scored_orders_by_score_and_keeps_top_k() -> None:
    reranker = CrossReranker("unused")
    reranker._model = FakeCrossEncoder([0.1, 0.9])
    chunks = [_chunk("a.py"), _chunk("b.py")]

    results = reranker.rerank_scored("question", chunks, top_k=1)

    assert [c.file_path for c, _ in results] == ["b.py"]
    assert results[0][1] == 0.9


def test_rerank_derives_from_rerank_scored() -> None:
    reranker = CrossReranker("unused")
    reranker._model = FakeCrossEncoder([0.1, 0.9])
    chunks = [_chunk("a.py"), _chunk("b.py")]

    results = reranker.rerank("question", chunks, top_k=2)

    assert [c.file_path for c in results] == ["b.py", "a.py"]


def test_rerank_keeps_low_scoring_chunks_when_the_top_one_is_strong() -> None:
    """The model scores "does this answer the question alone", so supporting context
    legitimately scores low. A strong top score means the question IS answerable, and
    those weak chunks are the context that makes the answer complete."""
    reranker = CrossReranker("unused", min_top_score=0.01)
    reranker._model = FakeCrossEncoder([0.884, 0.046, 0.003])
    chunks = [_chunk("a.py"), _chunk("b.py"), _chunk("c.py")]

    kept = reranker.rerank_scored("how does routing work", chunks, top_k=6)

    assert [c.file_path for c, _ in kept] == ["a.py", "b.py", "c.py"]


def test_rerank_rejects_everything_when_even_the_best_chunk_is_irrelevant() -> None:
    reranker = CrossReranker("unused", min_top_score=0.01)
    reranker._model = FakeCrossEncoder([0.0, 0.0])

    assert reranker.rerank_scored("who is the prime minister", [_chunk("a.py"), _chunk("b.py")], 6) == []


def test_rerank_still_honours_top_k_above_the_gate() -> None:
    reranker = CrossReranker("unused", min_top_score=0.01)
    reranker._model = FakeCrossEncoder([0.9, 0.8, 0.7, 0.6])
    chunks = [_chunk(f"{n}.py") for n in "abcd"]

    assert len(reranker.rerank_scored("q", chunks, top_k=2)) == 2


def test_mmr_prefers_a_diverse_lower_scoring_chunk_over_a_near_duplicate() -> None:
    """A (best score) and A' (near-duplicate direction of A, second-best score) vs B
    (lower score, orthogonal/genuinely different). Plain top-k would pick A, A' — two
    near-identical chunks. MMR should pick A, B instead: A' adds almost nothing new once
    A is already selected, while B covers different ground despite scoring lower."""
    reranker = CrossReranker("unused", min_top_score=0.01, mmr_lambda=0.5)
    reranker._model = FakeCrossEncoder([0.9, 0.85, 0.7])
    chunks = [_chunk("a.py"), _chunk("a_prime.py"), _chunk("b.py")]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    results = reranker.rerank_scored("q", chunks, top_k=2, vectors=vectors)

    assert [c.file_path for c, _ in results] == ["a.py", "b.py"]


def test_mmr_lambda_1_degenerates_to_plain_top_k() -> None:
    """mmr_lambda=1.0 means pure relevance, zero diversity weight — must produce the
    exact same selection as no-MMR top-k, not just something close to it."""
    reranker = CrossReranker("unused", min_top_score=0.01, mmr_lambda=1.0)
    reranker._model = FakeCrossEncoder([0.9, 0.85, 0.7])
    chunks = [_chunk("a.py"), _chunk("a_prime.py"), _chunk("b.py")]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    results = reranker.rerank_scored("q", chunks, top_k=2, vectors=vectors)

    assert [c.file_path for c, _ in results] == ["a.py", "a_prime.py"]


def test_mmr_still_rejects_everything_below_the_gate() -> None:
    reranker = CrossReranker("unused", min_top_score=0.01, mmr_lambda=0.5)
    reranker._model = FakeCrossEncoder([0.0, 0.0])

    results = reranker.rerank_scored(
        "who is the prime minister", [_chunk("a.py"), _chunk("b.py")], top_k=2,
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )

    assert results == []


def test_rerank_scored_with_diversity_flags_the_swapped_in_chunk() -> None:
    """A (best), A' (near-duplicate, 2nd best), B (lower score, diverse) — MMR swaps A'
    for B. B should be reported as a diversity pick since plain top-k wouldn't have
    included it; A should not, since it's the top scorer either way."""
    reranker = CrossReranker("unused", min_top_score=0.01, mmr_lambda=0.5)
    reranker._model = FakeCrossEncoder([0.9, 0.85, 0.7])
    chunks = [_chunk("a.py"), _chunk("a_prime.py"), _chunk("b.py")]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    results, diversity_picks = reranker.rerank_scored_with_diversity(
        "q", chunks, top_k=2, vectors=vectors
    )

    assert [c.file_path for c, _ in results] == ["a.py", "b.py"]
    assert diversity_picks == {"b.py::1-1"}  # only the swapped-in chunk, not the top scorer


def test_rerank_scored_with_diversity_is_empty_when_mmr_matches_plain_top_k() -> None:
    """No swap happened (MMR agreed with plain relevance order) — nothing to flag."""
    reranker = CrossReranker("unused", min_top_score=0.01, mmr_lambda=1.0)
    reranker._model = FakeCrossEncoder([0.9, 0.85, 0.7])
    chunks = [_chunk("a.py"), _chunk("a_prime.py"), _chunk("b.py")]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    _results, diversity_picks = reranker.rerank_scored_with_diversity(
        "q", chunks, top_k=2, vectors=vectors
    )

    assert diversity_picks == set()


def test_rerank_scored_with_diversity_is_empty_without_vectors() -> None:
    reranker = CrossReranker("unused", min_top_score=0.01)
    reranker._model = FakeCrossEncoder([0.9, 0.85, 0.7])
    chunks = [_chunk("a.py"), _chunk("a_prime.py"), _chunk("b.py")]

    _results, diversity_picks = reranker.rerank_scored_with_diversity("q", chunks, top_k=2)

    assert diversity_picks == set()


def test_rerank_scored_without_vectors_is_unaffected_by_mmr_lambda() -> None:
    """The no-vectors path (existing callers, evals, etc.) must stay exactly as it was —
    MMR only activates when vectors are actually supplied."""
    reranker = CrossReranker("unused", min_top_score=0.01, mmr_lambda=0.0)  # pure diversity, if it applied
    reranker._model = FakeCrossEncoder([0.9, 0.85, 0.7])
    chunks = [_chunk("a.py"), _chunk("a_prime.py"), _chunk("b.py")]

    results = reranker.rerank_scored("q", chunks, top_k=2)

    assert [c.file_path for c, _ in results] == ["a.py", "a_prime.py"]
