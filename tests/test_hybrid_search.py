from __future__ import annotations

from src.index.chunk_index import ChunkIndex
from src.index.schema import CodeChunk
from src.index.vector_store import VectorStore
from src.retrieve.hybrid_search import HybridSearch, RankFuser


def test_fuse_applies_weighted_min_max_normalization() -> None:
    fuser = RankFuser(dense_weight=1.0, bm25_weight=0.25)

    fused = fuser.fuse(dense_scores=[0.0, 1.0], bm25_scores=[0.0, 1.0])

    assert fused == [0.0, 1.25]


def test_fuse_guards_against_flat_score_lists() -> None:
    fuser = RankFuser(dense_weight=1.0, bm25_weight=1.0)

    fused = fuser.fuse(
        dense_scores=[5.0, 5.0],
        bm25_scores=[],
    )

    assert fused == []


def test_fuse_returns_zero_for_flat_scores() -> None:
    fuser = RankFuser(dense_weight=1.0, bm25_weight=1.0)

    fused = fuser.fuse(dense_scores=[3.0, 3.0], bm25_scores=[1.0, 2.0])

    assert fused == [0.0, 1.0]


def _chunk(file_path: str, code: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-1",
        space_id="demo",
        source_id="src1",
        file_path=file_path,
        language="python",
        symbol_name=None,
        start_line=1,
        end_line=1,
        code=code,
    )


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_one(self, text: str) -> list[float]:
        return self.vectors[text]


def test_search_fuses_dense_and_bm25_ranking() -> None:
    chunk_index = ChunkIndex(VectorStore(":memory:"))
    chunk_index.ensure(dim=2)
    chunk_index.upsert(
        [_chunk("a.py", "def parse_input(): pass"), _chunk("b.py", "def write_output(): pass")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    embedder = FakeEmbedder({"parse input": [1.0, 0.0]})
    search = HybridSearch(embedder, chunk_index, RankFuser(1.0, 0.25))

    results = search.search("parse input", space_id="demo", file_paths=[], k=1)

    assert [c.file_path for c in results] == ["a.py"]


def test_search_returns_empty_when_no_candidates() -> None:
    chunk_index = ChunkIndex(VectorStore(":memory:"))
    chunk_index.ensure(dim=2)
    search = HybridSearch(FakeEmbedder({}), chunk_index, RankFuser(1.0, 0.25))

    assert search.search("anything", space_id="demo", file_paths=[], k=5) == []


def test_search_scored_exposes_dense_bm25_and_fused_scores() -> None:
    chunk_index = ChunkIndex(VectorStore(":memory:"))
    chunk_index.ensure(dim=2)
    chunk_index.upsert(
        [_chunk("a.py", "def parse_input(): pass"), _chunk("b.py", "def write_output(): pass")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    embedder = FakeEmbedder({"parse input": [1.0, 0.0]})
    search = HybridSearch(embedder, chunk_index, RankFuser(1.0, 0.25))

    results = search.search_scored("parse input", space_id="demo", file_paths=[], k=2)

    assert [sc.chunk.file_path for sc in results] == ["a.py", "b.py"]
    assert results[0].dense_score > results[1].dense_score
    assert results[0].fused_score >= results[1].fused_score
    # The chunk's own stored vector rides along — not recomputed, just not discarded
    # anymore — so a later MMR diversity pass can compare candidates without a second
    # embedding call.
    assert results[0].vector == [1.0, 0.0]
    assert results[1].vector == [0.0, 1.0]
