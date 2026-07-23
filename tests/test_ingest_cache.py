from __future__ import annotations

from pathlib import Path

from src.ingest_cache import IngestCache


def test_load_returns_none_for_uncached_repo(tmp_path: Path) -> None:
    cache = IngestCache(str(tmp_path))

    assert cache.load("demo") is None


def test_save_then_load_round_trips_the_trace(tmp_path: Path) -> None:
    cache = IngestCache(str(tmp_path))
    trace = {"repo": "demo", "files": [{"file_path": "a.py"}]}

    cache.save("demo", trace)

    assert cache.load("demo") == trace


def test_list_repos_returns_sorted_cached_repo_names(tmp_path: Path) -> None:
    cache = IngestCache(str(tmp_path))
    cache.save("zeta", {"repo": "zeta"})
    cache.save("alpha", {"repo": "alpha"})

    assert cache.list_repos() == ["alpha", "zeta"]


def test_list_repos_returns_empty_list_when_cache_dir_missing(tmp_path: Path) -> None:
    cache = IngestCache(str(tmp_path / "does-not-exist"))

    assert cache.list_repos() == []
