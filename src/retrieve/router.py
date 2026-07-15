from __future__ import annotations

from src.index.doc_index import DocIndex
from src.index.embedder import Embedder


class Router:
    """Hierarchical routing: narrows a question to a shortlist of files before chunk search."""

    def __init__(self, embedder: Embedder, doc_index: DocIndex):
        self.embedder = embedder
        self.doc_index = doc_index

    def route_to_files(self, question: str, repo: str, top_files: int) -> list[str]:
        vector = self.embedder.embed_one(question)
        summaries = self.doc_index.search(vector, top_files, repo)
        return [s.file_path for s in summaries]
