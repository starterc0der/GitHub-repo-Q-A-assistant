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

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.repo}::{chunk.id}")),
                vector=vector,
                payload=self._to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.store.upsert(COLLECTION_NAME, points)

    def search(
        self,
        vector: list[float],
        limit: int,
        repo: str,
        file_paths: list[str] | None = None,
    ) -> list[CodeChunk]:
        results = self.store.search(
            COLLECTION_NAME, vector, limit, query_filter=self._filter(repo, file_paths)
        )
        return [self._to_chunk(point) for point in results]

    def fetch_by_files(
        self, repo: str, file_paths: list[str]
    ) -> list[tuple[CodeChunk, list[float]]]:
        """All chunks (with their stored vectors) for a set of already-routed files.

        Used by hybrid search: BM25 and dense fusion both need the same bounded
        candidate universe, scoped to files the router already picked.
        """
        records = self.store.scroll(
            COLLECTION_NAME,
            query_filter=self._filter(repo, file_paths),
            limit=10_000,
            with_vectors=True,
        )
        return [(self._to_chunk(record), record.vector) for record in records]

    def _filter(self, repo: str, file_paths: list[str] | None) -> Filter:
        must = [qmodels.FieldCondition(key="repo", match=qmodels.MatchValue(value=repo))]
        if file_paths:
            must.append(
                qmodels.FieldCondition(key="file_path", match=qmodels.MatchAny(any=file_paths))
            )
        return Filter(must=must)

    def _to_payload(self, chunk: CodeChunk) -> dict[str, object]:
        return {
            "chunk_id": chunk.id,
            "repo": chunk.repo,
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
            repo=payload["repo"],
            file_path=payload["file_path"],
            language=payload["language"],
            symbol_name=payload["symbol_name"],
            start_line=payload["start_line"],
            end_line=payload["end_line"],
            code=payload["code"],
            context_header=payload["context_header"],
        )
