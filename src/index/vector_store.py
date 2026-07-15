from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Filter


class VectorStore:
    """Thin Qdrant wrapper — same class serves ':memory:' tests and a real Docker instance.

    Deliberately payload-agnostic: callers (ChunkIndex, DocIndex) own the mapping
    between their domain objects and Qdrant points/payloads.
    """

    def __init__(self, url: str):
        self.client = QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url)

    def ensure_collection(self, name: str, dim: int) -> None:
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    def upsert(self, name: str, points: list[qmodels.PointStruct]) -> None:
        self.client.upsert(collection_name=name, points=points)

    def search(
        self,
        name: str,
        vector: list[float],
        limit: int,
        query_filter: Filter | None = None,
    ) -> list[qmodels.ScoredPoint]:
        return self.client.query_points(
            collection_name=name,
            query=vector,
            limit=limit,
            query_filter=query_filter,
        ).points

    def scroll(
        self,
        name: str,
        query_filter: Filter | None = None,
        limit: int = 1000,
        with_vectors: bool = False,
    ) -> list[qmodels.Record]:
        records, _ = self.client.scroll(
            collection_name=name,
            scroll_filter=query_filter,
            limit=limit,
            with_vectors=with_vectors,
        )
        return records
