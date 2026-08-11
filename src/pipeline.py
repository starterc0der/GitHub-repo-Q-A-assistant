from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.cancellation import Cancelled, check
from src.config import Settings
from src.generate.answer import SYSTEM_PROMPT, AnswerGenerator, CitationParser
from src.generate.intent import BroadIntentClassifier, matches_broad_keywords
from src.generate.rewriter import StandaloneRewriter
from src.index.chunk_index import ChunkIndex
from src.index.doc_index import DocIndex
from src.index.embedder import Embedder
from src.index.schema import CodeChunk, FileSummary
from src.index.vector_store import VectorStore
from src.ingest.ast_chunker import ASTChunker
from src.ingest.chunker import LANGUAGE_BY_EXT, RecursiveChunker, Tokenizer
from src.ingest.contextualizer import Contextualizer
from src.ingest.repo_loader import RepoLoader
from src.ingest.sources import chunk_csv, load_csv, load_docx, load_pdf, load_text
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
    VectorsTrace,
    WalkTrace,
    project_3d,
    vector_preview,
)

logger = logging.getLogger(__name__)

README_NAMES = ("README.md", "README.rst", "README.txt", "README")


@dataclass
class IngestReport:
    source_id: str
    name: str
    file_count: int
    chunk_count: int


class Pipeline:
    """Owns one instance of each pipeline stage and wires the full ingest/query flows.

    Ingest, per source kind:
      repo         clone -> walk -> AST-chunk + summarize -> contextualize -> embed -> index
      pdf/docx/text load -> chunk (line-based) -> template-summarize -> contextualize -> embed -> index
    Query: (standalone question) embed -> route -> hybrid search -> cross-rerank -> compress
           -> answer (raw question + chat history + compressed chunks).
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        self.repo_loader = RepoLoader()
        tokenizer = Tokenizer()
        self.recursive_chunker = RecursiveChunker(
            tokenizer, settings.chunk_max_chars, settings.chunk_overlap
        )
        self.ast_chunker = ASTChunker(
            tokenizer, self.recursive_chunker, settings.chunk_max_chars, settings.chunk_overlap
        )

        # Bulk client: one call per file and per chunk at ingest time, plus the cheap
        # standalone-question rewrite and broad-intent classification at chat time.
        bulk_llm = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_bulk_model)
        llm = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
        self.bulk_llm = bulk_llm

        self.summarizer = Summarizer(bulk_llm)
        self.contextualizer = Contextualizer(bulk_llm)
        self.rewriter = StandaloneRewriter(bulk_llm)
        self.intent_classifier = BroadIntentClassifier(bulk_llm)

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
    def default_source_name(url: str) -> str:
        """The display name derived from a clone URL when the caller doesn't supply one."""
        return Path(url).stem

    # ---------------------------------------------------------------- ingest

    def ingest_source(
        self, space_id: str, source_id: str, kind: str, name: str,
        uri: str | None = None, text: str | None = None,
    ) -> IngestReport:
        for event in self._run_ingest(space_id, source_id, kind, name, uri, text):
            if isinstance(event, tuple):
                return event[0]
        raise AssertionError("_run_ingest finished without yielding a result")

    def ingest_source_trace(
        self, space_id: str, source_id: str, kind: str, name: str,
        uri: str | None = None, text: str | None = None,
    ) -> IngestTrace:
        for event in self._run_ingest(space_id, source_id, kind, name, uri, text):
            if isinstance(event, tuple):
                return event[1]
        raise AssertionError("_run_ingest finished without yielding a result")

    def ingest_source_trace_stream(
        self, space_id: str, source_id: str, kind: str, name: str,
        uri: str | None = None, text: str | None = None,
    ) -> Iterator[IngestProgress | IngestTrace]:
        for event in self._run_ingest(space_id, source_id, kind, name, uri, text):
            yield event[1] if isinstance(event, tuple) else event

    def _run_ingest(
        self, space_id: str, source_id: str, kind: str, name: str,
        uri: str | None, text: str | None,
    ) -> Iterator[IngestProgress | tuple[IngestReport, IngestTrace]]:
        if kind == "repo":
            if uri is None:
                raise ValueError("repo sources require a uri")
            yield from self._run_ingest_repo(space_id, source_id, name, uri)
        else:
            yield from self._run_ingest_prose(space_id, source_id, kind, name, uri, text)

    def _run_ingest_repo(
        self, space_id: str, source_id: str, name: str, uri: str
    ) -> Iterator[IngestProgress | tuple[IngestReport, IngestTrace]]:
        yield IngestProgress(stage="clone", message=f"Cloning {name}…")
        dest = Path(self.settings.repo_clone_dir) / source_id
        root = self.repo_loader.clone(uri, dest)
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
            yield self._empty_result(space_id, source_id, "repo", name, uri, clone_trace, walk_trace)
            return

        yield IngestProgress(
            stage="summarize_repo", message="Summarizing repo…", files_total=len(paths)
        )
        readme = self._read_readme(root)
        file_tree = "\n".join(sorted(str(p.relative_to(root)) for p in paths))
        collection_summary = self.summarizer.summarize_repo(readme, file_tree)

        chunks: list[CodeChunk] = []
        file_summaries: list[FileSummary] = []
        chunk_spans: list[tuple[int, int]] = []
        for i, path in enumerate(paths):
            yield IngestProgress(
                stage="process_files",
                message=f"Chunking + summarizing {path.name}…",
                files_done=i,
                files_total=len(paths),
            )
            language = LANGUAGE_BY_EXT.get(path.suffix, "text")
            file_chunks = self.ast_chunker.chunk_file(path, root, space_id, source_id, language)
            if not file_chunks:
                continue

            file_path = str(path.relative_to(root))
            # dict.fromkeys dedupes while preserving order: every method chunk of a class
            # carries the class name, so a raw list repeats it once per method — noise in
            # the summarizer prompt and in the fallback summary built from it.
            symbols = list(dict.fromkeys(c.symbol_name for c in file_chunks if c.symbol_name))
            code = path.read_text(encoding="utf-8", errors="replace")
            file_summary = self.summarizer.summarize_file(
                space_id, source_id, file_path, language, code, symbols, collection_summary
            )
            file_summaries.append(file_summary)
            start = len(chunks)
            chunks.extend(
                self.contextualizer.add_context_headers_batch(
                    file_chunks, file_summary, collection_summary
                )
            )
            chunk_spans.append((start, len(chunks)))

        # No per-chunk progress is available inside embed()/upsert() — indeterminate
        # (files_total=0), same convention as clone/walk/summarize, instead of a fake 100%.
        yield IngestProgress(stage="embed", message="Embedding + indexing…")
        yield self._finish_ingest(
            space_id, source_id, "repo", name, uri, collection_summary,
            file_summaries, chunks, chunk_spans, clone_trace, walk_trace,
        )

    def _run_ingest_prose(
        self, space_id: str, source_id: str, kind: str, name: str,
        uri: str | None, text: str | None,
    ) -> Iterator[IngestProgress | tuple[IngestReport, IngestTrace]]:
        yield IngestProgress(stage="load", message=f"Loading {name}…")
        if kind == "pdf":
            files = load_pdf(Path(uri), name)
        elif kind == "docx":
            files = load_docx(Path(uri), name)
        elif kind == "text":
            files = load_text(text or "", name)
        elif kind == "csv":
            files = load_csv(Path(uri), name)
        else:
            raise ValueError(f"unknown source kind {kind!r}")

        if not files:
            yield self._empty_result(space_id, source_id, kind, name, uri, None, None)
            return

        chunks: list[CodeChunk] = []
        file_summaries: list[FileSummary] = []
        chunk_spans: list[tuple[int, int]] = []
        for i, logical_file in enumerate(files):
            yield IngestProgress(
                stage="process_files",
                message=f"Chunking {logical_file.name}…",
                files_done=i,
                files_total=len(files),
            )
            if kind == "csv":
                # Real summary, one call per CSV (not per chunk) — helps routing tell
                # different CSVs apart better than raw column names alone.
                file_summary = self.summarizer.summarize_csv(
                    space_id, source_id, logical_file.name, logical_file.text
                )
                file_chunks = chunk_csv(logical_file.text, logical_file.name, space_id, source_id)
                # chunk_csv already stamped the header into context_header directly —
                # Contextualizer would only overwrite it with a generic template.
            else:
                file_summary = self.summarizer.template_summary(
                    space_id, source_id, logical_file.name, "text", logical_file.text
                )
                file_chunks = self.recursive_chunker.chunk_text(
                    logical_file.text, logical_file.name, space_id, source_id, "text"
                )
                # Prose skips the LLM header (Contextualizer templates it) — see quick-win note.
                file_chunks = self.contextualizer.add_context_headers_batch(file_chunks, file_summary, "")
            if not file_chunks:
                continue
            file_summaries.append(file_summary)
            start = len(chunks)
            chunks.extend(file_chunks)
            chunk_spans.append((start, len(chunks)))

        # No per-chunk progress is available inside embed()/upsert() — indeterminate
        # (files_total=0), same convention as clone/walk/summarize, instead of a fake 100%.
        yield IngestProgress(stage="embed", message="Embedding + indexing…")
        yield self._finish_ingest(
            space_id, source_id, kind, name, uri, "", file_summaries, chunks, chunk_spans, None, None
        )

    def _empty_result(
        self, space_id: str, source_id: str, kind: str, name: str, uri: str | None,
        clone_trace: CloneTrace | None, walk_trace: WalkTrace | None,
    ) -> tuple[IngestReport, IngestTrace]:
        return IngestReport(source_id=source_id, name=name, file_count=0, chunk_count=0), IngestTrace(
            space_id=space_id, source_id=source_id, name=name, kind=kind, uri=uri,
            clone=clone_trace, walk=walk_trace,
        )

    def _finish_ingest(
        self, space_id: str, source_id: str, kind: str, name: str, uri: str | None,
        collection_summary: str, file_summaries: list[FileSummary], chunks: list[CodeChunk],
        chunk_spans: list[tuple[int, int]], clone_trace: CloneTrace | None, walk_trace: WalkTrace | None,
    ) -> tuple[IngestReport, IngestTrace]:
        if not chunks:
            return self._empty_result(space_id, source_id, kind, name, uri, clone_trace, walk_trace)

        # A source is re-ingested wholesale: drop its old chunks first so files removed
        # upstream (or an entirely re-parsed PDF) don't leave orphaned chunks behind.
        self.doc_index.ensure(self.embedder.dim)
        self.doc_index.delete_source(space_id, source_id)
        summary_vectors = self.embedder.embed([s.summary for s in file_summaries])
        self.doc_index.upsert(file_summaries, summary_vectors)

        self.chunk_index.ensure(self.embedder.dim)
        self.chunk_index.delete_source(space_id, source_id)
        chunk_vectors = self.embedder.embed([c.embeddable_text for c in chunks])
        self.chunk_index.upsert(chunks, chunk_vectors)

        logger.info(
            "Ingested %s (%s): %d files, %d chunks", name, kind, len(file_summaries), len(chunks)
        )

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
        report = IngestReport(
            source_id=source_id, name=name, file_count=len(file_summaries), chunk_count=len(chunks)
        )
        trace = IngestTrace(
            space_id=space_id, source_id=source_id, name=name, kind=kind, uri=uri,
            summary=collection_summary, clone=clone_trace, walk=walk_trace, files=files_trace,
        )
        return report, trace

    def delete_source(self, space_id: str, source_id: str) -> None:
        self.chunk_index.delete_source(space_id, source_id)
        self.doc_index.delete_source(space_id, source_id)

    def delete_space(self, space_id: str) -> None:
        self.chunk_index.delete_space(space_id)
        self.doc_index.delete_space(space_id)

    # ----------------------------------------------------------------- query

    # Nothing cleared the reranker's relevance floor, so there is no grounded answer to
    # generate. Returning this directly beats sending an empty context to the LLM: same
    # outcome, one fewer call, and it can't hallucinate its way around the gap.
    NO_MATCH = (
        "That question does not appear to relate to anything in this space — nothing in "
        "the indexed sources was relevant enough to answer it. If you meant something "
        "specific, try naming a file, page, or topic you're looking for."
    )

    def rewrite_standalone(self, question: str, history: list[tuple[str, str]]) -> tuple[bool, str]:
        """One cheap LLM call that resolves history-dependent references ("there", "it")
        into a self-contained question — the only thing retrieval ever sees. Skipped
        entirely when history is empty, so single-shot questions pay nothing extra.

        Returns (is_meta, text): is_meta means the question is about the conversation
        itself ("what have we discussed") rather than the document — see answer_meta."""
        return self.rewriter.rewrite(question, history)

    # "What have we talked about" is a question about the CONVERSATION, not the document —
    # the main answer prompt forbids using anything but retrieved chunks (so it always
    # refuses these), and retrieval itself would just burn a request trying to match
    # document content that was never the point. Bypasses retrieval entirely; bulk model,
    # since this is a plain recall/summary over a few chat turns, not deep reasoning.
    META_CHAT_SYSTEM_PROMPT = (
        "You are answering a question about THIS CONVERSATION so far — not about any "
        "document. Answer using only the conversation history provided; do not claim or "
        "invent anything about a source document beyond what was already said in this "
        "chat. When asked what topics, people, or items were discussed, list only the "
        "ones that were the actual subject of a question or a direct answer — not every "
        "name that happened to appear in passing inside an answer's supporting detail. "
        "Be concise."
    )

    def answer_meta(self, question: str, history: list[tuple[str, str]]) -> str:
        return self.bulk_llm.complete(question, system=self.META_CHAT_SYSTEM_PROMPT, history=history)

    def _wide_fallback_chunks(self, space_id: str, source_ids: list[str]) -> list[CodeChunk]:
        """Every chunk in the routed source(s), whole — for a question no single chunk can
        answer on its own (e.g. "summarize this"). Scoped to whole SOURCES rather than the
        routed FILES: routing caps at top_files, so a document with more pages/files than
        that cap would otherwise have its later pages silently missing from the "whole"
        answer — reading the entire source(s) instead removes that cap."""
        if not source_ids:
            return []
        return self.chunk_index.fetch_by_sources(space_id, source_ids)

    def _too_large_message(self, file_count: int, token_estimate: int) -> str:
        return (
            f"Answering that would need the full content of {file_count} file(s) "
            f"(~{token_estimate:,} tokens), which is too large to send in one request "
            f"(limit: {self.settings.wide_answer_max_tokens:,} tokens). Try asking about a "
            "specific file, section, or page instead."
        )

    def _answer_trace(
        self, question: str, chunks: list[CodeChunk], history: list[tuple[str, str]] | None,
        empty_message: str | None = None,
    ) -> AnswerTrace:
        """The answer call, wrapped so a failed generation still returns a viewable trace —
        every retrieval stage before it succeeded and is worth showing."""
        if not chunks:
            return AnswerTrace(
                text=empty_message or self.NO_MATCH, confidence=0.0, model=self.settings.llm_model
            )
        check()
        try:
            answer = self.answer_generator.answer(question, chunks, history=history)
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
            chart=answer.chart,
        )

    def query_trace(
        self, question: str, space_id: str,
        raw_question: str | None = None, history: list[tuple[str, str]] | None = None,
    ) -> QueryTrace:
        """Runs the same retrieval stages as query(), returning every stage's intermediate
        scores/chunks alongside the final answer. Logs each stage as it completes."""
        started = time.monotonic()
        stage = "embed-question"

        def done(step: str, detail: str) -> None:
            logger.info("  [%s] %s (%.1fs elapsed)", step, detail, time.monotonic() - started)

        logger.info("RETRIEVAL START | space=%s | question=%r", space_id, question)
        try:
            query_vector = self.embedder.embed_one(question)
            done("1/7 embed", f"{len(query_vector)}d query vector")

            stage = "route"
            routed = self.router.route_to_files_scored(
                question, space_id, self.settings.top_files, query_vector=query_vector
            )
            file_paths = [file_path for file_path, _source_id, _score in routed]
            # Only the top-scoring source(s), not every source with even one weak match —
            # Qdrant sorts score-descending, so routed[0] is the best.
            routed_source_ids: list[str] = []
            if routed:
                top_score = routed[0][2]
                routed_source_ids = list(dict.fromkeys(
                    source_id for _fp, source_id, score in routed if score == top_score
                ))
            done("2/7 route", f"{len(routed)} files shortlisted")

            stage = "hybrid-search"
            scored_candidates = self.hybrid_search.search_scored(
                question, space_id, file_paths, self.settings.hybrid_candidate_k,
                query_vector=query_vector,
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
            wide_fallback = False
            wide_reason = ""
            too_large_message: str | None = None
            check()

            # Three ways into the wide path, cheapest first — see BroadIntentClassifier's
            # docstring for why the LLM tier only fires on the genuinely ambiguous case.
            if not reranked_chunks:
                wide_fallback, wide_reason = True, "no chunk answered this on its own"
            elif matches_broad_keywords(question):
                wide_fallback, wide_reason = True, "question language implies broad coverage"
            elif self.intent_classifier.is_broad(question):
                wide_fallback, wide_reason = True, "classified as a broad-coverage question"

            if not wide_fallback:
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
                # falls back to the whole reranked list when compression dropped everything
                final_chunks = final_chunks or reranked_chunks
            else:
                # Skip compression — there's nothing to trim to, we want everything — and
                # gate only on total size, since sending it all is the point.
                wide_chunks = self._wide_fallback_chunks(space_id, routed_source_ids)
                wide_file_count = len({c.file_path for c in wide_chunks})
                token_estimate = (
                    sum(len(c.embeddable_text) for c in wide_chunks) // Tokenizer.CHARS_PER_TOKEN
                )
                if wide_chunks and token_estimate <= self.settings.wide_answer_max_tokens:
                    logger.info(
                        "  [5/7 compress] going wide (%s) — sending %d source(s), "
                        "%d whole file(s) (~%d tokens) instead",
                        wide_reason, len(routed_source_ids), wide_file_count, token_estimate,
                    )
                    final_chunks = wide_chunks
                    final_chunk_traces = [
                        CompressedChunkTrace(
                            chunk=chunk,
                            original_line_count=len(chunk.code.splitlines()),
                            compressed_line_count=len(chunk.code.splitlines()),
                            dropped=False,
                        )
                        for chunk in wide_chunks
                    ]
                elif wide_chunks:
                    logger.info(
                        "  [5/7 compress] going wide (%s) but whole-source fallback (~%d "
                        "tokens) exceeds the %d token budget — refusing",
                        wide_reason, token_estimate, self.settings.wide_answer_max_tokens,
                    )
                    too_large_message = self._too_large_message(wide_file_count, token_estimate)
                else:
                    logger.info(
                        "  [5/7 compress] going wide (%s) but no routed sources to fall back to",
                        wide_reason,
                    )

            done("6/7 prompt", f"{len(final_chunks)} chunks in the final prompt")
            stage = "answer"
        except Cancelled:
            logger.warning(
                "RETRIEVAL CANCELLED at stage '%s' after %.1fs | question=%r",
                stage, time.monotonic() - started, question,
            )
            raise

        answer_trace = self._answer_trace(
            raw_question or question, final_chunks, history, empty_message=too_large_message
        )
        logger.info(
            "RETRIEVAL DONE in %.1fs | %d citations | conf %.1f",
            time.monotonic() - started, len(answer_trace.citations), answer_trace.confidence,
        )

        return QueryTrace(
            question=question,
            space_id=space_id,
            query_embedding=vector_preview(query_vector),
            routed_files=[RoutedFile(file_path=fp, score=score) for fp, _source_id, score in routed],
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
            wide_fallback=wide_fallback,
            wide_fallback_reason=wide_reason,
            system_prompt=SYSTEM_PROMPT,
            final_prompt=self.answer_generator.build_prompt(raw_question or question, final_chunks),
            history=[{"role": role, "content": content} for role, content in (history or [])],
            answer=answer_trace,
        )

    def vectors_trace(self, question: str, space_id: str) -> VectorsTrace:
        """On-demand PCA projection for the vector-space UI — split out of query_trace()
        because it's two full-space scrolls + two SVDs that only matter if the user opens
        the pipeline-breakdown view, not something every message should pay for.

        `question` should be the same (standalone) question query_trace() embedded —
        embedding is deterministic, so re-embedding it here reproduces the same
        query_vector used at answer time."""
        query_vector = self.embedder.embed_one(question)

        all_file_vectors = self.doc_index.all_vectors(space_id)
        file_xyz_all = project_3d([query_vector] + [v for _, v in all_file_vectors])
        query_file_xyz, file_xyz_rest = file_xyz_all[0], file_xyz_all[1:]
        file_xyz = {
            summary.file_path: xyz for (summary, _), xyz in zip(all_file_vectors, file_xyz_rest)
        }

        # Every chunk in the space, not just a routed-file pool — shared by the
        # hybrid/rerank/compress plots and the "whole vector space" modal, so the query
        # lands in the same spot in both instead of two differently-scoped projections.
        whole_chunk_pool = self.chunk_index.fetch_by_files(space_id, [])
        whole_chunk_xyz_all = project_3d([query_vector] + [v for _, v in whole_chunk_pool])
        query_whole_chunk_xyz = whole_chunk_xyz_all[0]
        whole_chunk_xyz_rest = whole_chunk_xyz_all[1:]
        whole_chunk_xyz = {
            chunk.id: xyz for (chunk, _), xyz in zip(whole_chunk_pool, whole_chunk_xyz_rest)
        }
        chunk_labels = {
            chunk.id: f"{chunk.file_path} · {chunk.symbol_name or 'block'}"
            for chunk, _ in whole_chunk_pool
        }

        return VectorsTrace(
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
