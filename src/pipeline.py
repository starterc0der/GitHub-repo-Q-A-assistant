from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings
from src.generate.answer import SYSTEM_PROMPT, Answer, AnswerGenerator, CitationParser
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
from src.trace import (
    ChunkTrace,
    CloneTrace,
    CompressedChunkTrace,
    FileTrace,
    IngestProgress,
    IngestTrace,
    QueryTrace,
    RerankedChunkTrace,
    RoutedFile,
    ScoredChunkTrace,
    WalkTrace,
    project_2d,
    vector_preview,
)

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

    @staticmethod
    def repo_name(url: str) -> str:
        """The repo identifier derived from a clone URL — used both to name the local
        clone/index rows and as the cache key for a completed ingest trace."""
        return Path(url).stem

    def ingest_repo(self, url: str) -> IngestReport:
        for event in self._run_ingest(url):
            if isinstance(event, tuple):
                return event[0]
        raise AssertionError("_run_ingest finished without yielding a result")

    def ingest_repo_trace(self, url: str) -> IngestTrace:
        """Runs the same ingest as ingest_repo(), but returns per-file/per-chunk detail
        (summaries, context headers, embedding previews) instead of just counts."""
        for event in self._run_ingest(url):
            if isinstance(event, tuple):
                return event[1]
        raise AssertionError("_run_ingest finished without yielding a result")

    def ingest_repo_trace_stream(self, url: str) -> Iterator[IngestProgress | IngestTrace]:
        """Same ingest as ingest_repo_trace(), but yields IngestProgress updates along the
        way — powers the ingestion progress bar for large repos, where the per-file
        summarize/contextualize LLM calls can otherwise run for minutes with no feedback."""
        for event in self._run_ingest(url):
            yield event[1] if isinstance(event, tuple) else event

    def _run_ingest(
        self, url: str
    ) -> Iterator[IngestProgress | tuple[IngestReport, IngestTrace]]:
        repo = self.repo_name(url)
        yield IngestProgress(stage="clone", message=f"Cloning {repo}…")
        dest = Path(self.settings.repo_clone_dir) / repo
        root = self.repo_loader.clone(url, dest)
        clone_trace = CloneTrace(
            depth=1, commit=self.repo_loader.commit_sha(root), local_path=str(root)
        )

        yield IngestProgress(stage="walk", message="Scanning files…")
        walk = self.repo_loader.walk_report(root)
        walk_trace = WalkTrace(
            total_scanned=walk.total_scanned, kept=len(walk.kept), skipped=walk.skipped
        )
        paths = walk.kept
        if not paths:
            yield IngestReport(repo=repo, file_count=0, chunk_count=0), IngestTrace(
                repo=repo,
                repo_url=url,
                repo_summary="",
                clone=clone_trace,
                walk=walk_trace,
            )
            return

        yield IngestProgress(
            stage="summarize_repo", message="Summarizing repo…", files_total=len(paths)
        )
        readme = self._read_readme(root)
        file_tree = "\n".join(sorted(str(p.relative_to(root)) for p in paths))
        repo_summary = self.summarizer.summarize_repo(readme, file_tree)

        chunks: list[CodeChunk] = []
        file_summaries: list[FileSummary] = []
        chunk_spans: list[tuple[int, int]] = []  # per-file (start, end) index range into chunks
        for i, path in enumerate(paths):
            yield IngestProgress(
                stage="process_files",
                message=f"Chunking + summarizing {path.name}…",
                files_done=i,
                files_total=len(paths),
            )
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
            start = len(chunks)
            chunks.extend(
                self.contextualizer.add_context_header(chunk, file_summary, repo_summary)
                for chunk in file_chunks
            )
            chunk_spans.append((start, len(chunks)))

        if not chunks:
            yield IngestReport(repo=repo, file_count=0, chunk_count=0), IngestTrace(
                repo=repo,
                repo_url=url,
                repo_summary=repo_summary,
                clone=clone_trace,
                walk=walk_trace,
            )
            return

        yield IngestProgress(
            stage="embed",
            message="Embedding + indexing…",
            files_done=len(paths),
            files_total=len(paths),
        )
        self.doc_index.ensure(self.embedder.dim)
        summary_vectors = self.embedder.embed([s.summary for s in file_summaries])
        self.doc_index.upsert(file_summaries, summary_vectors)

        self.chunk_index.ensure(self.embedder.dim)
        chunk_vectors = self.embedder.embed([c.embeddable_text for c in chunks])
        self.chunk_index.upsert(chunks, chunk_vectors)

        logger.info("Ingested %s: %d files, %d chunks", repo, len(file_summaries), len(chunks))

        # One shared PCA space across every chunk in this ingest, so chunks from the
        # same file land close together in the vector-space visualization.
        chunk_xy = project_2d(chunk_vectors)

        files_trace = [
            FileTrace(
                file_path=file_summary.file_path,
                language=file_summary.language,
                summary=file_summary.summary,
                symbols=file_summary.symbols,
                summary_embedding=vector_preview(summary_vectors[i]),
                chunks=[
                    ChunkTrace(
                        chunk=chunks[j], embedding=vector_preview(chunk_vectors[j]), xy=chunk_xy[j]
                    )
                    for j in range(*chunk_spans[i])
                ],
            )
            for i, file_summary in enumerate(file_summaries)
        ]
        report = IngestReport(repo=repo, file_count=len(file_summaries), chunk_count=len(chunks))
        trace = IngestTrace(
            repo=repo,
            repo_url=url,
            repo_summary=repo_summary,
            clone=clone_trace,
            walk=walk_trace,
            files=files_trace,
        )
        yield report, trace

    def query(self, question: str, repo: str) -> Answer:
        file_paths = self.router.route_to_files(question, repo, self.settings.top_files)
        candidates = self.hybrid_search.search(
            question, repo, file_paths, self.settings.hybrid_candidate_k
        )
        reranked = self.cross_reranker.rerank(question, candidates, self.settings.rerank_top_k)

        compressed = [self.compressor.compress(question, chunk) for chunk in reranked]
        final_chunks = [chunk for chunk in compressed if chunk is not None] or reranked

        return self.answer_generator.answer(question, final_chunks)

    def query_trace(self, question: str, repo: str) -> QueryTrace:
        """Runs the same retrieval stages as query(), but stops before the answer LLM
        call and returns every stage's intermediate scores/chunks instead of an Answer."""
        query_vector = self.embedder.embed_one(question)
        routed = self.router.route_to_files_scored(question, repo, self.settings.top_files)
        file_paths = [file_path for file_path, _ in routed]

        scored_candidates = self.hybrid_search.search_scored(
            question, repo, file_paths, self.settings.hybrid_candidate_k
        )
        candidate_chunks = [sc.chunk for sc in scored_candidates]

        reranked_scored = self.cross_reranker.rerank_scored(
            question, candidate_chunks, self.settings.rerank_top_k
        )

        reranked_chunks = [chunk for chunk, _ in reranked_scored]
        final_chunk_traces: list[CompressedChunkTrace] = []
        final_chunks: list[CodeChunk] = []
        for chunk, _ in reranked_scored:
            compressed = self.compressor.compress(question, chunk)
            final_chunk_traces.append(
                CompressedChunkTrace(
                    chunk=compressed if compressed is not None else chunk,
                    original_line_count=len(chunk.code.splitlines()),
                    compressed_line_count=len(compressed.code.splitlines()) if compressed else 0,
                    dropped=compressed is None,
                )
            )
            if compressed is not None:
                final_chunks.append(compressed)
        # mirrors query()'s `final_chunks = [...] or reranked` fallback when compression
        # dropped everything
        final_chunks = final_chunks or reranked_chunks

        # Two PCA spaces for the vector-space visualization — see QueryTrace's field
        # comments for why they're separate.
        all_file_vectors = self.doc_index.all_vectors(repo)
        file_xy_all = project_2d([query_vector] + [v for _, v in all_file_vectors])
        query_file_xy, file_xy_rest = file_xy_all[0], file_xy_all[1:]
        file_xy = {
            summary.file_path: xy for (summary, _), xy in zip(all_file_vectors, file_xy_rest)
        }

        # Every chunk in the repo, not just the routed-file pool — shared by the
        # hybrid/rerank/compress plots and the "whole vector space" modal, so the query
        # lands in the same spot in both instead of two differently-scoped projections.
        whole_chunk_pool = self.chunk_index.fetch_by_files(repo, [])
        whole_chunk_xy_all = project_2d([query_vector] + [v for _, v in whole_chunk_pool])
        query_whole_chunk_xy, whole_chunk_xy_rest = whole_chunk_xy_all[0], whole_chunk_xy_all[1:]
        whole_chunk_xy = {
            chunk.id: xy for (chunk, _), xy in zip(whole_chunk_pool, whole_chunk_xy_rest)
        }
        # Full CodeChunk objects (whole_chunk_pool) are already in memory here — reuse them
        # for a cheap label per chunk rather than sending the whole chunk (code included)
        # for every one of what can be hundreds of repo-wide entries.
        chunk_labels = {
            chunk.id: f"{chunk.file_path} · {chunk.symbol_name or 'block'}"
            for chunk, _ in whole_chunk_pool
        }

        return QueryTrace(
            question=question,
            repo=repo,
            query_embedding=vector_preview(query_vector),
            routed_files=[RoutedFile(file_path=fp, score=score) for fp, score in routed],
            candidates=[
                ScoredChunkTrace(
                    chunk=sc.chunk,
                    dense_score=sc.dense_score,
                    bm25_score=sc.bm25_score,
                    fused_score=sc.fused_score,
                )
                for sc in scored_candidates
            ],
            reranked=[
                RerankedChunkTrace(chunk=chunk, rerank_score=score)
                for chunk, score in reranked_scored
            ],
            final_chunks=final_chunk_traces,
            system_prompt=SYSTEM_PROMPT,
            final_prompt=self.answer_generator.build_prompt(question, final_chunks),
            query_file_xy=query_file_xy,
            file_xy=file_xy,
            chunk_labels=chunk_labels,
            query_whole_chunk_xy=query_whole_chunk_xy,
            whole_chunk_xy=whole_chunk_xy,
        )

    def _read_readme(self, root: Path) -> str:
        for name in README_NAMES:
            path = root / name
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        return ""
