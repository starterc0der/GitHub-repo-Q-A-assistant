from __future__ import annotations

import uuid

from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Filter

from src.index.schema import FileSummary
from src.index.vector_store import VectorStore

COLLECTION_NAME = "file_summaries"


class DocIndex:
    """Owns the file-summary Qdrant collection used by the routing layer."""

    def __init__(self, store: VectorStore):
        self.store = store

    def ensure(self, dim: int) -> None:
        self.store.ensure_collection(COLLECTION_NAME, dim)

    def upsert(self, summaries: list[FileSummary], vectors: list[list[float]]) -> None:
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{s.repo}::{s.file_path}")),
                vector=vector,
                payload=self._to_payload(s),
            )
            for s, vector in zip(summaries, vectors)
        ]
        self.store.upsert(COLLECTION_NAME, points)

    def search(self, vector: list[float], limit: int, repo: str) -> list[FileSummary]:
        repo_filter = Filter(
            must=[qmodels.FieldCondition(key="repo", match=qmodels.MatchValue(value=repo))]
        )
        results = self.store.search(COLLECTION_NAME, vector, limit, query_filter=repo_filter)
        return [self._to_summary(point) for point in results]

    def _to_payload(self, s: FileSummary) -> dict[str, object]:
        return {
            "repo": s.repo,
            "file_path": s.file_path,
            "language": s.language,
            "summary": s.summary,
            "symbols": s.symbols,
        }

    def _to_summary(self, point: qmodels.ScoredPoint) -> FileSummary:
        payload = point.payload
        return FileSummary(
            repo=payload["repo"],
            file_path=payload["file_path"],
            language=payload["language"],
            summary=payload["summary"],
            symbols=payload["symbols"],
        )
