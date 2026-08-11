from __future__ import annotations

from src.index.doc_index import DocIndex
from src.index.embedder import Embedder


class Router:
    """Hierarchical routing: narrows a question to a shortlist of files before chunk search."""

    def __init__(self, embedder: Embedder, doc_index: DocIndex):
        self.embedder = embedder
        self.doc_index = doc_index

    def route_to_files(self, question: str, space_id: str, top_files: int) -> list[str]:
        vector = self.embedder.embed_one(question)
        summaries = self.doc_index.search(vector, top_files, space_id)
        return [s.file_path for s in summaries]

    def route_to_files_scored(
        self, question: str, space_id: str, top_files: int,
        query_vector: list[float] | None = None,
    ) -> list[tuple[str, str, float]]:
        """(file_path, source_id, score) per routed file. source_id is carried through
        (not just file_path) so a caller can expand from "these routed files" to "every
        file in the source(s) they belong to" — e.g. a whole-document fallback that must
        not stay capped at top_files. Pass query_vector when the caller already embedded
        the question, to avoid a redundant embedding forward pass."""
        vector = query_vector if query_vector is not None else self.embedder.embed_one(question)
        scored = self.doc_index.search_scored(vector, top_files, space_id)
        return [(s.file_path, s.source_id, score) for s, score in scored]
