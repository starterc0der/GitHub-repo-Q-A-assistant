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
        self.store.ensure_payload_index(COLLECTION_NAME, "space_id")
        self.store.ensure_payload_index(COLLECTION_NAME, "source_id")

    def upsert(self, summaries: list[FileSummary], vectors: list[list[float]]) -> None:
        points = [
            qmodels.PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL, f"{s.space_id}::{s.source_id}::{s.file_path}"
                    )
                ),
                vector=vector,
                payload=self._to_payload(s),
            )
            for s, vector in zip(summaries, vectors)
        ]
        self.store.upsert(COLLECTION_NAME, points)

    def delete_source(self, space_id: str, source_id: str) -> None:
        self.store.delete(COLLECTION_NAME, self._filter(space_id, source_id=source_id))

    def delete_space(self, space_id: str) -> None:
        self.store.delete(COLLECTION_NAME, self._filter(space_id))

    def search(self, vector: list[float], limit: int, space_id: str) -> list[FileSummary]:
        return [summary for summary, _ in self.search_scored(vector, limit, space_id)]

    def search_scored(
        self, vector: list[float], limit: int, space_id: str
    ) -> list[tuple[FileSummary, float]]:
        results = self.store.search(
            COLLECTION_NAME, vector, limit, query_filter=self._filter(space_id)
        )
        return [(self._to_summary(point), point.score) for point in results]

    def all_vectors(self, space_id: str) -> list[tuple[FileSummary, list[float]]]:
        """Every file summary + its vector for a space — used to build the routing-stage
        vector-space visualization, which needs the whole corpus, not just top-k matches."""
        records = self.store.scroll(
            COLLECTION_NAME, query_filter=self._filter(space_id), limit=10_000, with_vectors=True
        )
        return [(self._to_summary(record), record.vector) for record in records]

    def _filter(self, space_id: str, source_id: str | None = None) -> Filter:
        must = [qmodels.FieldCondition(key="space_id", match=qmodels.MatchValue(value=space_id))]
        if source_id:
            must.append(
                qmodels.FieldCondition(key="source_id", match=qmodels.MatchValue(value=source_id))
            )
        return Filter(must=must)

    def _to_payload(self, s: FileSummary) -> dict[str, object]:
        return {
            "space_id": s.space_id,
            "source_id": s.source_id,
            "file_path": s.file_path,
            "language": s.language,
            "summary": s.summary,
            "symbols": s.symbols,
        }

    def _to_summary(self, point: qmodels.ScoredPoint | qmodels.Record) -> FileSummary:
        payload = point.payload
        return FileSummary(
            space_id=payload["space_id"],
            source_id=payload["source_id"],
            file_path=payload["file_path"],
            language=payload["language"],
            summary=payload["summary"],
            symbols=payload["symbols"],
        )
