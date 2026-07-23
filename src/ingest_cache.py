from __future__ import annotations

import json
from pathlib import Path


class IngestCache:
    """Caches a completed ingest trace (the JSON-ready dict the API returns) per repo on
    disk. The chunk/summary vectors already persist in Qdrant after a successful ingest —
    this just saves the *display* data (summaries, context headers, embedding previews,
    walk stats) so re-selecting an already-ingested repo replays it instantly instead of
    re-cloning, re-summarizing, and re-embedding everything.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    def _path(self, repo: str) -> Path:
        return self.cache_dir / f"{repo}.json"

    def load(self, repo: str) -> dict[str, object] | None:
        path = self._path(repo)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, repo: str, trace: dict[str, object]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path(repo).write_text(json.dumps(trace), encoding="utf-8")

    def list_repos(self) -> list[str]:
        if not self.cache_dir.is_dir():
            return []
        return sorted(p.stem for p in self.cache_dir.glob("*.json"))
