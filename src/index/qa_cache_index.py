from __future__ import annotations

import uuid

from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Filter

from src.index.vector_store import VectorStore

COLLECTION_NAME = "qa_cache_questions"


class QaCacheIndex:
    """Owns the semantic-cache Qdrant collection — a small corpus of PAST turn-1
    QUESTIONS (not document chunks), searched to catch a paraphrase of an
    already-answered question. Exact-match (the SQLite `qa_cache` table) is tried
    first and is strictly cheaper/safer; this is the fallback for a miss there.

    Score thresholding is deliberately NOT done here — this class only returns the
    single closest match and its raw score. Pipeline.semantic_cache_lookup applies
    settings.semantic_cache_min_score, since "how conservative" is a policy decision,
    not something a thin index wrapper should bake in.
    """

    def __init__(self, store: VectorStore):
        self.store = store

    def ensure(self, dim: int) -> None:
        self.store.ensure_collection(COLLECTION_NAME, dim)
        self.store.ensure_payload_index(COLLECTION_NAME, "space_id")

    def upsert(
        self, space_id: str, question_hash: str, question: str, message_id: str, vector: list[float]
    ) -> None:
        point = qmodels.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{space_id}::{question_hash}")),
            vector=vector,
            payload={"space_id": space_id, "question": question, "message_id": message_id},
        )
        self.store.upsert(COLLECTION_NAME, [point])

    def delete_space(self, space_id: str) -> None:
        self.store.delete(COLLECTION_NAME, self._filter(space_id))

    def search_best(self, vector: list[float], space_id: str) -> tuple[str, str, float] | None:
        """Returns (message_id, matched_question, score) for the closest previously
        cached question in this space, or None if this space has cached nothing yet."""
        results = self.store.search(COLLECTION_NAME, vector, limit=1, query_filter=self._filter(space_id))
        if not results:
            return None
        point = results[0]
        return point.payload["message_id"], point.payload["question"], point.score

    def _filter(self, space_id: str) -> Filter:
        return Filter(must=[qmodels.FieldCondition(key="space_id", match=qmodels.MatchValue(value=space_id))])
