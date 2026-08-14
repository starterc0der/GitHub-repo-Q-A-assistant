from __future__ import annotations

from src.index.qa_cache_index import QaCacheIndex
from src.index.vector_store import VectorStore


def test_search_best_returns_closest_match_in_space() -> None:
    index = QaCacheIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert("demo", "hash-a", "who is Bhishma", "msg-a", [1.0, 0.0])
    index.upsert("demo", "hash-b", "who is Kripa", "msg-b", [0.0, 1.0])

    message_id, question, score = index.search_best([1.0, 0.0], "demo")

    assert (message_id, question) == ("msg-a", "who is Bhishma")
    assert score > 0.9


def test_search_best_is_scoped_by_space() -> None:
    index = QaCacheIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert("space-a", "hash-a", "who is Bhishma", "msg-a", [1.0, 0.0])

    assert index.search_best([1.0, 0.0], "space-b") is None


def test_search_best_returns_none_when_nothing_cached_yet() -> None:
    index = QaCacheIndex(VectorStore(":memory:"))
    index.ensure(dim=2)

    assert index.search_best([1.0, 0.0], "demo") is None


def test_upsert_same_hash_overwrites_rather_than_duplicates() -> None:
    """Re-caching the identical question (e.g. re-answering after invalidation) must
    replace the old vector, not accumulate a second point for the same question."""
    index = QaCacheIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert("demo", "hash-a", "who is Bhishma", "msg-old", [1.0, 0.0])
    index.upsert("demo", "hash-a", "who is Bhishma", "msg-new", [1.0, 0.0])

    message_id, _question, _score = index.search_best([1.0, 0.0], "demo")

    assert message_id == "msg-new"


def test_delete_space_removes_only_that_space() -> None:
    index = QaCacheIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert("space-a", "hash-a", "who is Bhishma", "msg-a", [1.0, 0.0])
    index.upsert("space-b", "hash-b", "who is Bhishma", "msg-b", [1.0, 0.0])

    index.delete_space("space-a")

    assert index.search_best([1.0, 0.0], "space-a") is None
    assert index.search_best([1.0, 0.0], "space-b") is not None
