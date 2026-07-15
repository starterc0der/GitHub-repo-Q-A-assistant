from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings
from src.generate.answer import Answer, AnswerGenerator, CitationParser
from src.index.chunk_index import ChunkIndex
from src.index.doc_index import DocIndex
from src.index.embedder import Embedder
from src.index.schema import CodeChunk, FileSummary
from src.index.vector_store import VectorStore
from src.ingest.ast_chunker import ASTChunker
from src.ingest.chunker import LANGUAGE_BY_EXT, RecursiveChunker, Tokenizer
from src.ingest.contextualizer import Contextualizer
from src.ingest.repo_loader import RepoLoader
from src.ingest.summarizer import Summarizer
from src.llm_client import LLMClient
from src.retrieve.compressor import Compressor
from src.retrieve.cross_reranker import CrossReranker
from src.retrieve.hybrid_search import HybridSearch, RankFuser
from src.retrieve.router import Router

logger = logging.getLogger(__name__)

README_NAMES = ("README.md", "README.rst", "README.txt", "README")


@dataclass
class IngestReport:
    repo: str
    file_count: int
    chunk_count: int


class Pipeline:
    """Owns one instance of each pipeline stage and wires the full ingest/query flows.

    Ingest: clone -> walk -> AST-chunk + summarize -> contextualize -> embed -> index.
    Query: embed -> route to files -> hybrid search -> cross-rerank -> compress -> answer.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        self.repo_loader = RepoLoader()
        tokenizer = Tokenizer()
        self.ast_chunker = ASTChunker(
            tokenizer,
            RecursiveChunker(tokenizer, settings.chunk_max_chars, settings.chunk_overlap),
            settings.chunk_max_chars,
            settings.chunk_overlap,
        )

        llm = LLMClient(settings)
        self.summarizer = Summarizer(llm)
        self.contextualizer = Contextualizer(llm)

        self.embedder = Embedder(settings.embedding_model)
        store = VectorStore(settings.qdrant_url)
        self.chunk_index = ChunkIndex(store)
        self.doc_index = DocIndex(store)

        self.router = Router(self.embedder, self.doc_index)
        self.hybrid_search = HybridSearch(
            self.embedder,
            self.chunk_index,
            RankFuser(settings.hybrid_dense_weight, settings.hybrid_bm25_weight),
        )
        self.cross_reranker = CrossReranker(settings.reranker_model)
        self.compressor = Compressor(llm)
        self.answer_generator = AnswerGenerator(llm, CitationParser())

    def ingest_repo(self, url: str) -> IngestReport:
        repo = Path(url).stem
        dest = Path(self.settings.repo_clone_dir) / repo
        root = self.repo_loader.clone(url, dest)
        paths = list(self.repo_loader.walk_files(root))
        if not paths:
            return IngestReport(repo=repo, file_count=0, chunk_count=0)

        readme = self._read_readme(root)
        file_tree = "\n".join(sorted(str(p.relative_to(root)) for p in paths))
        repo_summary = self.summarizer.summarize_repo(readme, file_tree)

        chunks: list[CodeChunk] = []
        file_summaries: list[FileSummary] = []
        for path in paths:
            language = LANGUAGE_BY_EXT.get(path.suffix, "text")
            file_chunks = self.ast_chunker.chunk_file(path, root, repo, language)
            if not file_chunks:
                continue

            file_path = str(path.relative_to(root))
            symbols = [c.symbol_name for c in file_chunks if c.symbol_name]
            code = path.read_text(encoding="utf-8", errors="replace")
            file_summary = self.summarizer.summarize_file(
                repo, file_path, language, code, symbols, repo_summary
            )
            file_summaries.append(file_summary)
            chunks.extend(
                self.contextualizer.add_context_header(chunk, file_summary, repo_summary)
                for chunk in file_chunks
            )

        if not chunks:
            return IngestReport(repo=repo, file_count=0, chunk_count=0)

        self.doc_index.ensure(self.embedder.dim)
        summary_vectors = self.embedder.embed([s.summary for s in file_summaries])
        self.doc_index.upsert(file_summaries, summary_vectors)

        self.chunk_index.ensure(self.embedder.dim)
        chunk_vectors = self.embedder.embed([c.embeddable_text for c in chunks])
        self.chunk_index.upsert(chunks, chunk_vectors)

        logger.info("Ingested %s: %d files, %d chunks", repo, len(file_summaries), len(chunks))
        return IngestReport(repo=repo, file_count=len(file_summaries), chunk_count=len(chunks))

    def query(self, question: str, repo: str) -> Answer:
        file_paths = self.router.route_to_files(question, repo, self.settings.top_files)
        candidates = self.hybrid_search.search(
            question, repo, file_paths, self.settings.hybrid_candidate_k
        )
        reranked = self.cross_reranker.rerank(question, candidates, self.settings.rerank_top_k)

        compressed = [self.compressor.compress(question, chunk) for chunk in reranked]
        final_chunks = [chunk for chunk in compressed if chunk is not None] or reranked

        return self.answer_generator.answer(question, final_chunks)

    def _read_readme(self, root: Path) -> str:
        for name in README_NAMES:
            path = root / name
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        return ""
