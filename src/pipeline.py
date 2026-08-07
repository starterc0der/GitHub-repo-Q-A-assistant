from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.cancellation import Cancelled, check
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
    AnswerTrace,
    ChunkTrace,
    CitationTrace,
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
    project_3d,
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

        # Bulk client: one call per file and per chunk at ingest time.
        bulk_llm = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_bulk_model)
        llm = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)

        self.summarizer = Summarizer(bulk_llm)
        self.contextualizer = Contextualizer(bulk_llm)

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
        self.cross_reranker = CrossReranker(settings.reranker_model, settings.rerank_min_top_score)
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
            # dict.fromkeys dedupes while preserving order: every method chunk of a class
            # carries the class name, so a raw list repeats it once per method — noise in
            # the summarizer prompt and in the fallback summary built from it.
            symbols = list(dict.fromkeys(c.symbol_name for c in file_chunks if c.symbol_name))
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
        chunk_xyz = project_3d(chunk_vectors)

        files_trace = [
            FileTrace(
                file_path=file_summary.file_path,
                language=file_summary.language,
                summary=file_summary.summary,
                symbols=file_summary.symbols,
                summary_embedding=vector_preview(summary_vectors[i]),
                chunks=[
                    ChunkTrace(
                        chunk=chunks[j], embedding=vector_preview(chunk_vectors[j]), xyz=chunk_xyz[j]
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

    # Nothing cleared the reranker's relevance floor, so there is no grounded answer to
    # generate. Returning this directly beats sending an empty context to the LLM: same
    # outcome, one fewer call, and it can't hallucinate its way around the gap.
    NO_MATCH = (
        "That question does not appear to relate to this repository — nothing in the "
        "indexed code was relevant enough to answer it. If you meant something in this "
        "codebase, try naming a file, function, or behaviour you are looking for."
    )

    def query(self, question: str, repo: str) -> Answer:
        file_paths = self.router.route_to_files(question, repo, self.settings.top_files)
        candidates = self.hybrid_search.search(
            question, repo, file_paths, self.settings.hybrid_candidate_k
        )
        reranked = self.cross_reranker.rerank(question, candidates, self.settings.rerank_top_k)
        if not reranked:
            return Answer(text=self.NO_MATCH, citations=[], confidence=0.0)

        check()  # everything below here is LLM calls — don't start them for a dead caller
        compressed = self.compressor.compress_batch(question, reranked)
        final_chunks = [chunk for chunk in compressed if chunk is not None] or reranked

        return self.answer_generator.answer(question, final_chunks)

    def _answer_trace(self, question: str, chunks: list[CodeChunk]) -> AnswerTrace:
        """The answer call, wrapped so a failed generation still returns a viewable trace —
        every retrieval stage before it succeeded and is worth showing."""
        if not chunks:
            return AnswerTrace(text=self.NO_MATCH, confidence=0.0, model=self.settings.llm_model)
        check()
        try:
            answer = self.answer_generator.answer(question, chunks)
        except RuntimeError as exc:
            # confidence=0: AnswerTrace defaults to 1.0, which would render a failed
            # answer as a confident one.
            return AnswerTrace(
                text="", confidence=0.0, model=self.settings.llm_model, error=str(exc)
            )
        return AnswerTrace(
            text=answer.text,
            citations=[
                CitationTrace(
                    file_path=c.file_path,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    snippet=c.snippet,
                )
                for c in answer.citations
            ],
            confidence=answer.confidence,
            model=self.settings.llm_model,
        )

    def query_trace(self, question: str, repo: str) -> QueryTrace:
        """Runs the same retrieval stages as query(), returning every stage's intermediate
        scores/chunks alongside the final answer. Logs each stage as it completes."""
        started = time.monotonic()
        stage = "embed-question"

        def done(step: str, detail: str) -> None:
            logger.info("  [%s] %s (%.1fs elapsed)", step, detail, time.monotonic() - started)

        logger.info("RETRIEVAL START | repo=%s | question=%r", repo, question)
        try:
            query_vector = self.embedder.embed_one(question)
            done("1/7 embed", f"{len(query_vector)}d query vector")

            stage = "route"
            routed = self.router.route_to_files_scored(question, repo, self.settings.top_files)
            file_paths = [file_path for file_path, _ in routed]
            done("2/7 route", f"{len(routed)} files shortlisted")

            stage = "hybrid-search"
            scored_candidates = self.hybrid_search.search_scored(
                question, repo, file_paths, self.settings.hybrid_candidate_k
            )
            candidate_chunks = [sc.chunk for sc in scored_candidates]
            done("3/7 hybrid", f"{len(candidate_chunks)} candidates")

            stage = "rerank"
            reranked_scored = self.cross_reranker.rerank_scored(
                question, candidate_chunks, self.settings.rerank_top_k
            )
            done(
                "4/7 rerank",
                f"{len(reranked_scored)} kept of {len(candidate_chunks)}"
                + ("" if reranked_scored else " — top score below gate"),
            )

            stage = "compress"
            reranked_chunks = [chunk for chunk, _ in reranked_scored]
            final_chunk_traces: list[CompressedChunkTrace] = []
            final_chunks: list[CodeChunk] = []
            check()
            logger.info("  [5/7 compress] %d chunks in one batched call", len(reranked_chunks))
            compressed_chunks = self.compressor.compress_batch(question, reranked_chunks)
            for chunk, compressed in zip(reranked_chunks, compressed_chunks):
                final_chunk_traces.append(
                    CompressedChunkTrace(
                        chunk=compressed if compressed is not None else chunk,
                        original_line_count=len(chunk.code.splitlines()),
                        compressed_line_count=(
                            len(compressed.code.splitlines()) if compressed else 0
                        ),
                        dropped=compressed is None,
                    )
                )
                if compressed is not None:
                    final_chunks.append(compressed)
            # mirrors query()'s `final_chunks = [...] or reranked` fallback when
            # compression dropped everything
            final_chunks = final_chunks or reranked_chunks

            stage = "project"
            # Two PCA spaces for the vector-space visualization — see QueryTrace's field
            # comments for why they're separate.
            all_file_vectors = self.doc_index.all_vectors(repo)
            file_xyz_all = project_3d([query_vector] + [v for _, v in all_file_vectors])
            query_file_xyz, file_xyz_rest = file_xyz_all[0], file_xyz_all[1:]
            file_xyz = {
                summary.file_path: xyz
                for (summary, _), xyz in zip(all_file_vectors, file_xyz_rest)
            }

            # Every chunk in the repo, not just the routed-file pool — shared by the
            # hybrid/rerank/compress plots and the "whole vector space" modal, so the
            # query lands in the same spot in both instead of two differently-scoped
            # projections.
            whole_chunk_pool = self.chunk_index.fetch_by_files(repo, [])
            whole_chunk_xyz_all = project_3d([query_vector] + [v for _, v in whole_chunk_pool])
            query_whole_chunk_xyz = whole_chunk_xyz_all[0]
            whole_chunk_xyz_rest = whole_chunk_xyz_all[1:]
            whole_chunk_xyz = {
                chunk.id: xyz for (chunk, _), xyz in zip(whole_chunk_pool, whole_chunk_xyz_rest)
            }
            # Full CodeChunk objects (whole_chunk_pool) are already in memory here — reuse
            # them for a cheap label per chunk rather than sending the whole chunk (code
            # included) for every one of what can be hundreds of repo-wide entries.
            chunk_labels = {
                chunk.id: f"{chunk.file_path} · {chunk.symbol_name or 'block'}"
                for chunk, _ in whole_chunk_pool
            }

            done("6/7 prompt", f"{len(final_chunks)} chunks in the final prompt")
            stage = "answer"
        except Cancelled:
            logger.warning(
                "RETRIEVAL CANCELLED at stage '%s' after %.1fs | question=%r",
                stage, time.monotonic() - started, question,
            )
            raise

        answer_trace = self._answer_trace(question, final_chunks)
        logger.info(
            "RETRIEVAL DONE in %.1fs | %d citations | conf %.1f",
            time.monotonic() - started, len(answer_trace.citations), answer_trace.confidence,
        )

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
            rerank_min_top_score=self.settings.rerank_min_top_score,
            system_prompt=SYSTEM_PROMPT,
            final_prompt=self.answer_generator.build_prompt(question, final_chunks),
            answer=answer_trace,
            query_file_xyz=query_file_xyz,
            file_xyz=file_xyz,
            chunk_labels=chunk_labels,
            query_whole_chunk_xyz=query_whole_chunk_xyz,
            whole_chunk_xyz=whole_chunk_xyz,
        )

    def _read_readme(self, root: Path) -> str:
        for name in README_NAMES:
            path = root / name
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        return ""
