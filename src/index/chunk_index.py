from __future__ import annotations

import uuid

from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Filter

from src.index.schema import CodeChunk
from src.index.vector_store import VectorStore

COLLECTION_NAME = "code_chunks"


class ChunkIndex:
    """Owns the code-chunk Qdrant collection: CodeChunk <-> point/payload mapping."""

    def __init__(self, store: VectorStore):
        self.store = store

    def ensure(self, dim: int) -> None:
        self.store.ensure_collection(COLLECTION_NAME, dim)
        self.store.ensure_payload_index(COLLECTION_NAME, "space_id")
        self.store.ensure_payload_index(COLLECTION_NAME, "source_id")

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
        points = [
            qmodels.PointStruct(
                id=self._point_id(chunk),
                vector=vector,
                payload=self._to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.store.upsert(COLLECTION_NAME, points)

    def delete_source(self, space_id: str, source_id: str) -> None:
        """Delete every chunk for one source — called before re-ingest and on source
        removal, so stale chunks for files no longer upstream never linger."""
        self.store.delete(COLLECTION_NAME, self._filter(space_id, source_id=source_id))

    def delete_space(self, space_id: str) -> None:
        self.store.delete(COLLECTION_NAME, self._filter(space_id))

    def search(
        self,
        vector: list[float],
        limit: int,
        space_id: str,
        file_paths: list[str] | None = None,
    ) -> list[CodeChunk]:
        results = self.store.search(
            COLLECTION_NAME, vector, limit, query_filter=self._filter(space_id, file_paths)
        )
        return [self._to_chunk(point) for point in results]

    def fetch_by_files(
        self, space_id: str, file_paths: list[str]
    ) -> list[tuple[CodeChunk, list[float]]]:
        """All chunks (with their stored vectors) for a set of already-routed files.

        Used by hybrid search: BM25 and dense fusion both need the same bounded
        candidate universe, scoped to files the router already picked.
        """
        records = self.store.scroll(
            COLLECTION_NAME,
            query_filter=self._filter(space_id, file_paths),
            limit=10_000,
            with_vectors=True,
        )
        return [(self._to_chunk(record), record.vector) for record in records]

    def fetch_by_sources(self, space_id: str, source_ids: list[str]) -> list[CodeChunk]:
        """Every chunk belonging to any of the given sources — unlike fetch_by_files, not
        limited to whichever files routing's top_files cap let through, so a whole-document
        fallback reads the entire source instead of a routing-sized slice of it. No vectors:
        this is a whole-document read, not a similarity search."""
        if not source_ids:
            return []
        records = self.store.scroll(
            COLLECTION_NAME, query_filter=self._filter(space_id, source_ids=source_ids), limit=10_000
        )
        return [self._to_chunk(record) for record in records]

    def peek_source(self, space_id: str, source_id: str, limit: int) -> list[CodeChunk]:
        """A cheap, bounded sample of one source's chunks — for classifying what a source
        IS (e.g. "does this parse as a place doc?") without paying to read a large,
        irrelevant source in full just to rule it out."""
        records = self.store.scroll(
            COLLECTION_NAME, query_filter=self._filter(space_id, source_id=source_id), limit=limit
        )
        return [self._to_chunk(record) for record in records]

    def fetch_by_ids(self, chunk_ids: list[str]) -> list[CodeChunk]:
        """Rehydrate chunk bodies by their stable point id — used to redisplay a stored
        trace without re-embedding or re-storing the chunk body per message."""
        if not chunk_ids:
            return []
        records = self.store.retrieve(COLLECTION_NAME, chunk_ids)
        return [self._to_chunk(record) for record in records]

    def point_id(self, space_id: str, source_id: str, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{space_id}::{source_id}::{chunk_id}"))

    def _point_id(self, chunk: CodeChunk) -> str:
        return self.point_id(chunk.space_id, chunk.source_id, chunk.id)

    def _filter(
        self, space_id: str, file_paths: list[str] | None = None,
        source_id: str | None = None, source_ids: list[str] | None = None,
    ) -> Filter:
        must = [qmodels.FieldCondition(key="space_id", match=qmodels.MatchValue(value=space_id))]
        if source_id:
            must.append(
                qmodels.FieldCondition(key="source_id", match=qmodels.MatchValue(value=source_id))
            )
        if source_ids:
            must.append(
                qmodels.FieldCondition(key="source_id", match=qmodels.MatchAny(any=source_ids))
            )
        if file_paths:
            must.append(
                qmodels.FieldCondition(key="file_path", match=qmodels.MatchAny(any=file_paths))
            )
        return Filter(must=must)

    def _to_payload(self, chunk: CodeChunk) -> dict[str, object]:
        return {
            "chunk_id": chunk.id,
            "space_id": chunk.space_id,
            "source_id": chunk.source_id,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "symbol_name": chunk.symbol_name,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "code": chunk.code,
            "context_header": chunk.context_header,
        }

    def _to_chunk(self, point: qmodels.ScoredPoint | qmodels.Record) -> CodeChunk:
        payload = point.payload
        return CodeChunk(
            id=payload["chunk_id"],
            space_id=payload["space_id"],
            source_id=payload["source_id"],
            file_path=payload["file_path"],
            language=payload["language"],
            symbol_name=payload["symbol_name"],
            start_line=payload["start_line"],
            end_line=payload["end_line"],
            code=payload["code"],
            context_header=payload["context_header"],
        )
