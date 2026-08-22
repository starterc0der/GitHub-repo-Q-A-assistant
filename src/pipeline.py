from __future__ import annotations

import contextvars
import logging
import math
import threading
import time
from collections.abc import Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from datetime import date

from src.cancellation import Cancelled, check
from src.config import Settings
from src.connectors.live_data import (
    LIVE_DATA_SYSTEM_PROMPT,
    PlaceDevices,
    build_live_data_context,
    extract_table,
    fetch_live_readings,
    find_matching_places,
    list_space_source_ids,
    parse_place_blocks,
)
from src.connectors.reports import (
    ReportWindowResolver,
    build_report_block,
    build_report_context,
    fetch_report_data,
    infer_chart_kind,
    strip_date_phrases,
)
from src.generate.answer import SYSTEM_PROMPT, AnswerGenerator
from src.generate.decomposer import DecomposeResult, QueryDecomposer
from src.generate.faithfulness import FaithfulnessChecker, FaithfulnessResult
from src.generate.intent import BroadIntentClassifier, matches_broad_keywords
from src.generate.memory import ConversationMemory
from src.generate.provenance import ClaimAttributor, ClaimCitation
from src.generate.sufficiency import SufficiencyChecker
from src.generate.rewriter import StandaloneRewriter
from src.index.chunk_index import ChunkIndex
from src.index.doc_index import DocIndex
from src.index.embedder import Embedder
from src.index.qa_cache_index import QaCacheIndex
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
from src.retrieve.hybrid_search import HybridSearch, RankFuser, ScoredChunk
from src.retrieve.router import Router
from src.trace import (
    AnswerTrace,
    ChunkTrace,
    CloneTrace,
    CompressedChunkTrace,
    FileTrace,
    IngestProgress,
    IngestTrace,
    LiveDataToolTrace,
    QueryTrace,
    ReportToolTrace,
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


def _add_usage(totals: dict[str, int], usage: dict | None) -> None:
    if not usage:
        return
    totals["prompt_tokens"] = totals.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
    totals["completion_tokens"] = totals.get("completion_tokens", 0) + usage.get("completion_tokens", 0)


def _table_as_claims_text(table: dict) -> str:
    """Renders a live-data ```table block back into claim-shaped sentences for
    ClaimAttributor — the numeric readings live in the table now (see
    LIVE_DATA_SYSTEM_PROMPT's anti-duplication rule), so a faithfulness check that only
    looked at the prose `text` would verify little more than a one-line aside and never
    touch the actual pressure/flow/totalizer values a live-data answer exists to report.
    Only numeric cells become claims — a placeholder ("unavailable", "N/A", "") is the
    model honestly reporting a metric that doesn't apply here (e.g. "level" only exists
    on an inlet's reading, never an outlet's — see build_live_data_context), not a fact
    needing grounding, and checking it anyway flagged every one as unsupported since the
    context only ever states a metric's presence, never its absence."""
    lines = []
    for row in table["rows"]:
        pairs = ", ".join(
            f"{col}={val}" for col, val in zip(table["columns"][1:], row[1:])
            if isinstance(val, (int, float))
        )
        if pairs:
            lines.append(f"{row[0]}: {pairs}")
    return "\n".join(lines)


@dataclass
class IngestReport:
    source_id: str
    name: str
    file_count: int
    chunk_count: int


@dataclass
class _CandidateState:
    """Stages 1-4 (embed→route→hybrid→rerank) for ONE (sub-)question — split out so
    query decomposition can run this once per sub-question, in parallel, before the
    merge step and the shared compress/generate stages that follow."""

    query_vector: list[float]
    routed: list[tuple[str, str, float]]
    scored_candidates: list[ScoredChunk]
    reranked_scored: list[tuple[CodeChunk, float]]
    # Chunk IDs in reranked_scored that MMR picked over a higher-scoring alternative for
    # diversity — see CrossReranker.rerank_scored_with_diversity. Empty when MMR wasn't
    # used or happened to agree with plain top-k anyway.
    diversity_picks: set[str] = field(default_factory=set)
    # ms per stage: embed, route, hybrid, rerank.
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class _RetrievalState:
    """Everything route→hybrid→rerank→compress produced — enough to run generation
    (blocking or streaming) and, once the answer text is known, build the QueryTrace."""

    query_vector: list[float]
    routed: list[tuple[str, str, float]]
    scored_candidates: list[ScoredChunk]
    reranked_scored: list[tuple[CodeChunk, float]]
    final_chunks: list[CodeChunk]
    final_chunk_traces: list[CompressedChunkTrace]
    wide_fallback: bool
    wide_fallback_reason: str
    too_large_message: str | None
    diversity_picks: set[str] = field(default_factory=set)
    # Empty unless the question was decomposed; source_question maps below are then keyed
    # by file_path (routed) or chunk.id (candidate/reranked) -> which sub-question won it.
    sub_questions: list[str] = field(default_factory=list)
    routed_origin: dict[str, str] = field(default_factory=dict)
    candidate_origin: dict[str, str] = field(default_factory=dict)
    reranked_origin: dict[str, str] = field(default_factory=dict)
    # ms per stage (embed/route/hybrid/rerank from candidate, plus decompose/gate/compress
    # added here) and cumulative {prompt_tokens, completion_tokens} across the LLM calls
    # this retrieval made (decompose, broad-intent if it fired, compress).
    timings: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    sufficiency: str = "sufficient"
    insufficient_sub_questions: list[str] = field(default_factory=list)
    # original sub-question -> its retry rewrite, for every insufficient sub-question that
    # got one retry attempt (whether or not the retry found anything). Empty unless at
    # least one sub-question was insufficient after the first pass. See Pipeline._retrieve.
    retried_sub_questions: dict[str, str] = field(default_factory=dict)
    # "single" | "parallel" | "sequential" — see QueryDecomposer.decompose. Only changes
    # how sub_questions/timings.caption should be worded in the UI; every stage after
    # decompose treats sub_questions identically regardless of mode.
    decompose_mode: str = "single"
    # From the same upfront classification — is_broad already decided whether hybrid+
    # rerank ran at all (see _retrieve_candidates' skip_hybrid_rerank); wants_chart is
    # passed into the answer prompt as an explicit hint. See DecomposeResult.
    is_broad: bool = False
    wants_chart: bool = False
    wants_live_data: bool = False
    wants_report: bool = False
    # Every chunk of every source that content-classifies as a place/device doc, fetched
    # whole — only populated for a live-data/report question (see _finish_retrieval).
    # Deliberately bypasses file routing and the shared hybrid_candidate_k cap: both are
    # similarity-based and neither reliably surfaces a small device-catalog doc sharing
    # the space with a much larger, differently-domained corpus — see _finish_retrieval's
    # wants_tool_answer branch for how sources get classified instead.
    tool_source_chunks: list[CodeChunk] = field(default_factory=list)


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
        self.llm = llm

        self.summarizer = Summarizer(bulk_llm)
        self.contextualizer = Contextualizer(bulk_llm)
        self.rewriter = StandaloneRewriter(bulk_llm, settings.max_subquestions)
        self.intent_classifier = BroadIntentClassifier(bulk_llm)
        self.decomposer = QueryDecomposer(bulk_llm, settings.max_subquestions)
        self.report_window_resolver = ReportWindowResolver(bulk_llm)
        self.faithfulness_checker = FaithfulnessChecker(bulk_llm)
        self.claim_attributor = ClaimAttributor(bulk_llm)
        self.conversation_memory = ConversationMemory(bulk_llm)
        self.sufficiency_checker = SufficiencyChecker(bulk_llm)

        self.embedder = Embedder(settings.embedding_model)
        store = VectorStore(settings.qdrant_url)
        self.chunk_index = ChunkIndex(store)
        self.doc_index = DocIndex(store)
        self.qa_cache_index = QaCacheIndex(store)

        self.router = Router(self.embedder, self.doc_index)
        self.hybrid_search = HybridSearch(
            self.embedder,
            self.chunk_index,
            RankFuser(settings.hybrid_dense_weight, settings.hybrid_bm25_weight),
        )
        self.cross_reranker = CrossReranker(
            settings.reranker_model, settings.rerank_min_top_score, settings.mmr_lambda
        )
        self.compressor = Compressor(llm)
        self.answer_generator = AnswerGenerator(llm, max_context_tokens=settings.answer_context_max_tokens)

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
        self.qa_cache_index.delete_space(space_id)

    def invalidate_qa_cache(self, space_id: str) -> None:
        """A source finishing ingest or being removed can change the answer to any
        question in this space — called alongside the SQLite `qa_cache` wipe at both call
        sites (ingest completion, source deletion) so the semantic-cache vectors don't
        outlive the exact-match rows they're supposed to mirror."""
        self.qa_cache_index.delete_space(space_id)

    # ----------------------------------------------------------------- query

    # Nothing cleared the reranker's relevance floor, so there is no grounded answer to
    # generate. Returning this directly beats sending an empty context to the LLM: same
    # outcome, one fewer call, and it can't hallucinate its way around the gap.
    NO_MATCH = (
        "That question does not appear to relate to anything in this space — nothing in "
        "the indexed sources was relevant enough to answer it. If you meant something "
        "specific, try naming a file, page, or topic you're looking for."
    )

    def route_question(
        self, question: str, history: list[tuple[str, str]]
    ) -> tuple[DecomposeResult, float, dict[str, int]]:
        """The one classification call every turn makes before retrieval runs — decides
        what KIND of question this is (meta/broad/chart/single/parallel/sequential) so
        the rest of the pipeline can pick the right strategy instead of discovering it
        partway through. Turn 1 (no history) and turn 2+ (has history) use different LLM
        calls under the hood — QueryDecomposer vs. StandaloneRewriter, since only the
        latter needs to resolve history-dependent references ("there", "it") — but both
        speak the same grammar (see decomposer.ROUTE_GRAMMAR), so a question is
        classified identically regardless of which turn it's asked on.

        Returns (result, ms, tokens) — the caller passes ms/tokens straight into
        _retrieve() (see its decompose_ms/decompose_tokens params) so this call's cost
        still shows up in the trace instead of vanishing just because the caller, not
        _retrieve() itself, happened to make it.

        result.is_meta means the question is about the conversation/assistant itself
        rather than the document — see answer_meta. For a meta question,
        result.sub_questions[0] is just the original question, unused by the caller."""
        self.bulk_llm.last_usage = None
        started = time.monotonic()
        result = self.decomposer.decompose(question) if not history else self.rewriter.rewrite(question, history)
        ms = (time.monotonic() - started) * 1000
        tokens: dict[str, int] = {}
        _add_usage(tokens, self.bulk_llm.last_usage)
        return result, ms, tokens

    # Two different things share the is_meta bypass (see QueryDecomposer/StandaloneRewriter's
    # shared grammar): a greeting/capability question ("hi", "what can you do") — which can
    # happen turn 1, before any history exists — and "what have we talked about", which needs
    # real history. Both skip retrieval entirely: the main answer prompt forbids using
    # anything but retrieved chunks (so it would always refuse these), and retrieval itself
    # would just burn a request trying to match document content that was never the point.
    # Bulk model, since this is a plain greeting/recall/summary, not deep reasoning.
    META_CHAT_SYSTEM_PROMPT = (
        "You are answering a question about THIS CONVERSATION or about yourself as the "
        "assistant — not about any source document.\n"
        "If it's a greeting, thanks, or a question about what you can help with, answer "
        "briefly and naturally: you answer questions about the documents/sources in this "
        "space, grounded only in what's actually indexed there — invite them to ask "
        "something about the material.\n"
        "If it's a question about the conversation itself (what was discussed, who was "
        "mentioned, a summary/recap), answer using only the conversation history "
        "provided; do not claim or invent anything about a source document beyond what "
        "was already said in this chat. List only topics/people/items that were the "
        "actual subject of a question or a direct answer — not every name that happened "
        "to appear in passing inside an answer's supporting detail.\n"
        "If the history includes a leading message from a 'system' role, that is a "
        "compact summary of earlier turns that scrolled out of the raw window — treat "
        "everything in it exactly as if those turns were asked and answered normally, "
        "with the same weight as the turns after it. Never mention that a summary "
        "exists, that turns were condensed, or comment on the conversation's format — "
        "just answer using everything you were given, older and recent alike.\n"
        "Be concise."
    )

    def answer_meta(self, question: str, history: list[tuple[str, str]]) -> str:
        return self.bulk_llm.complete(question, system=self.META_CHAT_SYSTEM_PROMPT, history=history)

    def semantic_cache_lookup(self, space_id: str, question: str) -> tuple[str, str, float] | None:
        """Called only on an exact-match cache MISS (see chat_routes._cache_lookup):
        embeds `question` locally (no LLM call) and finds the closest previously-cached
        turn-1 question in this space. Applies settings.semantic_cache_min_score itself —
        "how conservative" is a policy decision that belongs with the setting, not left to
        every caller to remember. Returns (message_id, matched_question, score), or None
        if nothing is close enough (including "nothing cached yet")."""
        vector = self.embedder.embed_one(question)
        self.qa_cache_index.ensure(self.embedder.dim)
        best = self.qa_cache_index.search_best(vector, space_id)
        if best is None or best[2] < self.settings.semantic_cache_min_score:
            return None
        return best

    def semantic_cache_put(self, space_id: str, question_hash: str, question: str, message_id: str) -> None:
        vector = self.embedder.embed_one(question)
        self.qa_cache_index.ensure(self.embedder.dim)
        self.qa_cache_index.upsert(space_id, question_hash, question, message_id, vector)

    def check_faithfulness(
        self, question: str, answer_text: str, chunks: list[CodeChunk]
    ) -> FaithfulnessResult:
        """Not called on the live chat path — eval-only for now (see evals/runner.py).
        One extra bulk-LLM call; wiring it into chat_routes.py per-message is a separate
        cost/latency decision, not something this method decides on its own."""
        return self.faithfulness_checker.check(question, answer_text, chunks)

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
        empty_message: str | None = None, insufficient: list[str] | None = None,
        wants_chart: bool = False,
    ) -> AnswerTrace:
        """The answer call, wrapped so a failed generation still returns a viewable trace —
        every retrieval stage before it succeeded and is worth showing."""
        if not chunks:
            return AnswerTrace(text=empty_message or self.NO_MATCH, model=self.settings.llm_model)
        check()
        try:
            answer = self.answer_generator.answer(
                question, chunks, history=history, insufficient=insufficient, wants_chart=wants_chart
            )
        except RuntimeError as exc:
            return AnswerTrace(text="", model=self.settings.llm_model, error=str(exc))
        citations = self.claim_attributor.attribute(answer.text, chunks)
        return AnswerTrace(
            text=answer.text, model=self.settings.llm_model, chart=answer.chart, citations=citations
        )

    @staticmethod
    def _match_places_in_chunks(question: str, chunks: list[CodeChunk]) -> list[PlaceDevices]:
        """Shared first step of both _try_live_data_answer and _try_report_answer:
        parse every place/device block out of the given chunks (deduped, since a place
        can appear in more than one retrieved chunk) and pick which one(s) the question
        is actually about (see find_matching_places — a question can name more than one
        place, or a device id instead of a place name). Capped at 5 so a "give me
        everything" question doesn't blow up the context/table unboundedly. Empty list
        means "can't identify a place here" — callers fall back to the normal answer."""
        all_places: list[PlaceDevices] = []
        seen_chunk_ids: set[str] = set()
        for chunk in chunks:
            if chunk.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.id)
            all_places.extend(parse_place_blocks(chunk.code))
        if not all_places:
            return []
        return find_matching_places(question, all_places)[:5]

    def _try_live_data_answer(
        self, question: str, space_id: str, chunks: list[CodeChunk]
    ) -> tuple[str, dict | None, LiveDataToolTrace, list[ClaimCitation]] | None:
        """Answers a live-data question (see DecomposeResult.wants_live_data) directly
        from Redis instead of the normal chunk-grounded generation. `chunks` should be
        the final_chunks PLUS the raw hybrid-search candidates (see
        _live_data_candidate_chunks) — not final_chunks alone: file-level routing scores
        a whole source, and in a space mixing a small device doc with much larger
        unrelated sources, a razor-thin routing-score gap can send the generic
        wide-fallback path after the wrong (huge) source entirely, leaving final_chunks
        empty even though hybrid search's own per-chunk candidates found the right one
        fine. Fetches each matched place's live readings and narrates them.
        Returns (text, table, tool_trace, citations) — table is the parsed ```table block
        (see live_data.TableParser) when the model emitted one, else None; the answer
        prompt defaults to a table unless the question explicitly asked for something
        else. citations comes from the same ClaimAttributor used on the normal
        chunk-grounded path, run against a synthetic chunk wrapping the fetched-readings
        context instead of a real retrieved chunk — this is the only LLM-narrated tool
        answer (report answers are built in code, not generated), so it's the only one
        that can actually misstate a reading and needs checking.

        Returns None (never raises) for anything that means "can't answer this live" —
        no place-doc chunk among what was retrieved, no place name match, no Redis
        connector configured for this space, an unrecognized ULB — so the caller falls
        back to the normal chunk-grounded answer rather than the user getting nothing."""
        matches = self._match_places_in_chunks(question, chunks)
        if not matches:
            return None

        readings: dict[str, dict | None] = {}
        answerable = []
        for place in matches:
            place_readings = fetch_live_readings(space_id, place.ulb, place.points)
            if place_readings is None:
                continue  # this place's ULB/connector isn't available — skip it, don't fail the whole answer
            readings.update(place_readings)
            answerable.append(place)
        if not answerable:
            return None

        context = build_live_data_context(answerable, readings)
        try:
            reply = self.llm.complete(f"{context}\n\nQuestion: {question}", system=LIVE_DATA_SYSTEM_PROMPT)
        except RuntimeError:
            return None
        text, table = extract_table(reply)
        context_chunk = CodeChunk(
            id=f"live-data:{space_id}", space_id=space_id, source_id="",
            file_path="Live data tool (Redis)", language="text", symbol_name=None,
            start_line=1, end_line=context.count("\n") + 1, code=context,
        )
        # A table answer's real numeric content lives in `table`, not `text` (see
        # LIVE_DATA_SYSTEM_PROMPT's anti-duplication rule) — check both, or the readings
        # that are the actual point of the answer never get verified at all.
        checkable_text = f"{text}\n{_table_as_claims_text(table)}".strip() if table else text
        citations = self.claim_attributor.attribute(checkable_text, [context_chunk])
        tool_trace = LiveDataToolTrace(
            matched_places=[p.place_name for p in answerable],
            redis_keys=list(readings.keys()),
            readings=readings,
            context=context,
        )
        return text, table, tool_trace, citations

    def _try_report_answer(
        self, question: str, space_id: str, chunks: list[CodeChunk], wants_chart: bool = False,
    ) -> tuple[str, dict | None, dict | None, ReportToolTrace] | None:
        """Answers a historical-report question (see DecomposeResult.wants_report) from
        the watco_stream_db aggregate tables instead of the normal chunk-grounded
        generation — same place/device-doc matching as _try_live_data_answer, but
        resolves a time WINDOW first (see ReportWindowResolver) and fetches from
        Postgres instead of Redis. The answer text/table/chart is built directly from
        the fetched data (see build_report_block) rather than generated by an LLM call:
        that path was tried first and, even after a compliance-focused prompt and a
        retry, still unreliably dropped the required block or emitted malformed JSON
        on a long multi-series chart — confirmed live, repeatedly. Building it in code
        makes both failure modes structurally impossible instead of merely less likely.

        Returns None for anything that means "can't even attempt a report here" — no
        place-doc chunk among what was retrieved, no place name match, the time
        reference couldn't be understood, or no Postgres connector configured for this
        space — so the caller falls back to the normal chunk-grounded answer. It does
        NOT return None just because a matched place/device has zero rows in the report
        tables — that's a real, answerable case ("no report data available for X"),
        stated honestly by build_report_block instead of the caller silently describing
        the place's static catalog entry instead."""
        # See strip_date_phrases's own comment — a report question's date (unlike a
        # live-data question, which rarely has one) can otherwise false-match an
        # unrelated place whose name happens to contain the same day-of-month number.
        matches = self._match_places_in_chunks(strip_date_phrases(question), chunks)
        if not matches:
            return None

        window = self.report_window_resolver.resolve(question, date.today())
        if window is None:
            return None

        points = [p for place in matches for p in place.points]
        reports = fetch_report_data(space_id, points, window)
        if reports is None:
            return None

        context = build_report_context(matches, window, reports)
        chart_kind = infer_chart_kind(question)
        text, table, chart = build_report_block(matches, reports, window, wants_chart, chart_kind)
        tool_trace = ReportToolTrace(
            matched_places=[p.place_name for p in matches],
            metric=window.metric, granularity=window.granularity,
            start_date=window.start_date.isoformat(), end_date=window.end_date.isoformat(),
            context=context,
        )
        return text, table, chart, tool_trace

    @staticmethod
    def _live_data_candidate_chunks(state: _RetrievalState) -> list[CodeChunk]:
        """See _try_live_data_answer's/_try_report_answer's docstrings for why this is
        more than just final_chunks — the raw hybrid-search candidates carry chunks the
        wide-fallback file-selection heuristic can lose entirely. Shared by both tool
        paths (live-data and report), not live-data-specific despite the name.

        tool_source_chunks (the top-routed source(s) read whole — see
        _RetrievalState's comment) goes last and deduped by id: it's the completeness
        guarantee, not the primary source, so a chunk already present via
        scored_candidates isn't listed twice."""
        seen = {c.id for c in state.final_chunks}
        seen.update(sc.chunk.id for sc in state.scored_candidates)
        extra = [c for c in state.tool_source_chunks if c.id not in seen]
        return state.final_chunks + [sc.chunk for sc in state.scored_candidates] + extra

    def _retrieve_candidates(
        self, question: str, space_id: str, skip_hybrid_rerank: bool = False, skip_rerank: bool = False
    ) -> _CandidateState:
        """Stages 1-4: embed→route→hybrid search→cross-rerank, for ONE question. Called
        once directly, or once per sub-question (in parallel) when the question was
        decomposed — see _retrieve().

        skip_hybrid_rerank stops after routing, for a question the router already
        classified as broad (see DecomposeResult.is_broad) — hybrid+rerank exist to find
        the single best-matching chunk, which a "summarize this" question doesn't need,
        and rerank is the single most expensive stage in this pipeline. Leaves
        reranked_scored empty, which _finish_retrieval's existing "no chunk answered this
        on its own" branch already reads as "go wide" when routing found something
        relevant — so this needs no new downstream branching.

        skip_rerank runs hybrid search but skips only the cross-encoder step (the
        single most expensive stage — ~1s/candidate on this CPU) — for a live-data
        question (see DecomposeResult.wants_live_data), which never needed a
        confidently-reranked chunk: Pipeline._try_live_data_answer scans the raw hybrid
        candidates (scored_candidates) directly, unlike skip_hybrid_rerank which would
        also lose those."""
        started = time.monotonic()
        timings: dict[str, float] = {}
        stage_started = started

        def done(step: str, key: str, detail: str) -> None:
            nonlocal stage_started
            now = time.monotonic()
            timings[key] = (now - stage_started) * 1000
            logger.info("  [%s] %r %s (%.1fs elapsed)", step, question, detail, now - started)
            stage_started = now

        query_vector = self.embedder.embed_one(question)
        done("1/6 embed", "embed", f"{len(query_vector)}d query vector")

        routed = self.router.route_to_files_scored(
            question, space_id, self.settings.top_files, query_vector=query_vector
        )
        file_paths = [file_path for file_path, _source_id, _score in routed]
        done("2/6 route", "route", f"{len(routed)} files shortlisted")

        if skip_hybrid_rerank:
            return _CandidateState(query_vector, routed, [], [], timings=timings)

        scored_candidates = self.hybrid_search.search_scored(
            question, space_id, file_paths, self.settings.hybrid_candidate_k,
            query_vector=query_vector,
        )
        candidate_chunks = [sc.chunk for sc in scored_candidates]
        candidate_vectors = [sc.vector for sc in scored_candidates]
        done("3/6 hybrid", "hybrid", f"{len(candidate_chunks)} candidates")

        if skip_rerank:
            reranked_scored, diversity_picks = [], set()
            done("4/6 rerank", "rerank", "skipped — live-data question")
        else:
            reranked_scored, diversity_picks = self.cross_reranker.rerank_scored_with_diversity(
                question, candidate_chunks, self.settings.rerank_top_k, vectors=candidate_vectors
            )
            done(
                "4/6 rerank", "rerank",
                f"{len(reranked_scored)} kept of {len(candidate_chunks)}"
                + ("" if reranked_scored else " — top score below gate"),
            )
        return _CandidateState(
            query_vector, routed, scored_candidates, reranked_scored, diversity_picks, timings
        )

    HOP_CONTEXT_TOP_K = 3
    HOP_CONTEXT_CHUNK_CHARS = 150

    def _hop_context(self, state: _CandidateState) -> str:
        """A zero-extra-LLM-call stand-in for a hop's answer, fed to resolve_hop as the
        raw material it extracts a clean answer from — the top-K reranked chunks (not
        just the top-1) since the specific answer often lands a few slots down even when
        the top chunk is a good match for the question overall. Empty if the hop itself
        found nothing (see Pipeline._retrieve's sequential branch)."""
        if not state.reranked_scored:
            return ""
        parts = [
            " ".join(chunk.code.split())[: self.HOP_CONTEXT_CHUNK_CHARS]
            for chunk, _score in state.reranked_scored[: self.HOP_CONTEXT_TOP_K]
        ]
        return " ".join(parts)

    def _merge_candidates(
        self, sub_questions: list[str], states: list[_CandidateState]
    ) -> tuple[_CandidateState, dict[str, str], dict[str, str], dict[str, str]]:
        """Combines N independent candidate states (one per sub-question) into one.
        Routed files and hybrid candidates are unioned, best score wins. Reranked chunks
        take a fair share from EACH sub-question's own top-k instead of a flat top-k over
        pooled scores — cross-encoder scores aren't comparable across different queries,
        so pooling them would just reflect which sub-question happened to score "louder",
        silently dropping a whole sub-topic's chunks before compression ever sees them.

        Returns the merged state plus origin maps (file_path / chunk.id -> the winning
        sub-question) purely for the trace — retrieval itself doesn't need them."""
        if len(states) == 1:
            return states[0], {}, {}, {}

        query_vector = [sum(vals) / len(states) for vals in zip(*(s.query_vector for s in states))]

        routed_best: dict[str, tuple[str, str, float]] = {}
        routed_origin: dict[str, str] = {}
        for sub_q, state in zip(sub_questions, states):
            for file_path, source_id, score in state.routed:
                if file_path not in routed_best or score > routed_best[file_path][2]:
                    routed_best[file_path] = (file_path, source_id, score)
                    routed_origin[file_path] = sub_q
        routed = sorted(routed_best.values(), key=lambda r: r[2], reverse=True)

        candidate_best: dict[str, ScoredChunk] = {}
        candidate_origin: dict[str, str] = {}
        for sub_q, state in zip(sub_questions, states):
            for sc in state.scored_candidates:
                if sc.chunk.id not in candidate_best or sc.fused_score > candidate_best[sc.chunk.id].fused_score:
                    candidate_best[sc.chunk.id] = sc
                    candidate_origin[sc.chunk.id] = sub_q
        scored_candidates = sorted(candidate_best.values(), key=lambda sc: sc.fused_score, reverse=True)

        share = math.ceil(self.settings.rerank_top_k / len(states))
        reranked_best: dict[str, tuple[CodeChunk, float]] = {}
        reranked_origin: dict[str, str] = {}
        for sub_q, state in zip(sub_questions, states):
            for chunk, score in state.reranked_scored[:share]:
                if chunk.id not in reranked_best or score > reranked_best[chunk.id][1]:
                    reranked_best[chunk.id] = (chunk, score)
                    reranked_origin[chunk.id] = sub_q
        reranked_scored = sorted(reranked_best.values(), key=lambda cs: cs[1], reverse=True)
        diversity_picks: set[str] = set().union(*(s.diversity_picks for s in states))

        # Branches ran concurrently, so each stage's real contribution to wall-clock time
        # is however long the SLOWEST branch took at that stage, not the sum of all of them.
        timings = {
            key: max(s.timings.get(key, 0.0) for s in states)
            for key in ("embed", "route", "hybrid", "rerank")
        }

        merged = _CandidateState(
            query_vector, routed, scored_candidates, reranked_scored, diversity_picks, timings
        )
        return merged, routed_origin, candidate_origin, reranked_origin

    def _finish_retrieval(
        self, question: str, space_id: str, candidate: _CandidateState, sub_questions: list[str],
        routed_origin: dict[str, str], candidate_origin: dict[str, str],
        reranked_origin: dict[str, str], started: float,
        wants_live_data: bool = False, wants_report: bool = False,
    ) -> _RetrievalState:
        """Stages 5-6: broad-intent gate, then compress (or whole-source wide fallback).
        Runs once against the ORIGINAL question and the merged candidate/rerank lists —
        sub-questions exist only to fix retrieval; generation answers what was actually
        asked, not each fragment separately.

        wants_live_data/wants_report skip the wide-fallback gate entirely (see
        Pipeline._try_live_data_answer/_try_report_answer) — both are a targeted device
        lookup, never "summarize this source," and file-level routing scores a whole
        source: in a space mixing a small device doc with much larger unrelated
        sources, a razor-thin routing-score gap can send wide-fallback after the wrong
        (huge) source, blowing the token budget for a question that never needed it."""
        wants_tool_answer = wants_live_data or wants_report
        # Only the top-scoring source(s), not every source with even one weak match —
        # candidate.routed is score-descending, so routed[0] is the best.
        routed_source_ids: list[str] = []
        if candidate.routed:
            top_score = candidate.routed[0][2]
            routed_source_ids = list(dict.fromkeys(
                source_id for _fp, source_id, score in candidate.routed if score == top_score
            ))

        check()
        timings: dict[str, float] = dict(candidate.timings)
        tokens: dict[str, int] = {}
        gate_started = time.monotonic()
        reranked_chunks = [chunk for chunk, _ in candidate.reranked_scored]
        final_chunk_traces: list[CompressedChunkTrace] = []
        final_chunks: list[CodeChunk] = []
        wide_fallback = False
        wide_reason = ""
        too_large_message: str | None = None

        # Three ways into the wide path, cheapest first — see BroadIntentClassifier's
        # docstring for why the LLM tier only fires on the genuinely ambiguous case.
        route_top_score = candidate.routed[0][2] if candidate.routed else 0.0
        tool_source_chunks: list[CodeChunk] = []
        if wants_tool_answer:
            logger.info("  [5/6 compress] live-data/report question — skipping wide fallback entirely")
            # File-level routing (DocIndex.search_scored) is dense-embedding-only — no
            # lexical/BM25 signal — so a short, structured device-catalog doc routinely
            # loses to hundreds of unrelated prose pages in a mixed corpus: confirmed
            # live, "puri device data" missed the top-8 routed files outright on every
            # phrasing tried, and "ctc device data" missed them whenever the question
            # didn't happen to also name "Cuttack". Tool-answer place matching only
            # needs a handful of small, deterministically-identifiable device-catalog
            # sources, so this bypasses routing entirely for this path: peek at each
            # source's first few chunks (cheap even for a 217-page PDF — one bounded
            # fetch, not the whole source) and keep only the ones that actually parse as
            # a place doc, rather than trusting a similarity score to have found them.
            tool_token_estimate = 0
            for source_id in list_space_source_ids(space_id):
                preview = self.chunk_index.peek_source(space_id, source_id, limit=3)
                if not parse_place_blocks("\n".join(c.code for c in preview)):
                    continue
                source_chunks = self._wide_fallback_chunks(space_id, [source_id])
                source_tokens = sum(len(c.embeddable_text) for c in source_chunks) // Tokenizer.CHARS_PER_TOKEN
                if tool_token_estimate + source_tokens > self.settings.wide_answer_max_tokens:
                    logger.info(
                        "  [5/6 compress] tool-source %s too large (~%d tokens) — skipping it, "
                        "keeping other tool sources", source_id, source_tokens,
                    )
                    continue
                tool_source_chunks.extend(source_chunks)
                tool_token_estimate += source_tokens
        elif not reranked_chunks and route_top_score < self.settings.route_min_top_score:
            # Off-topic, not broad — skip the wide attempt so this reports NO_MATCH
            # instead of a "too large, be more specific" refusal.
            logger.info(
                "  [5/6 compress] no reranked chunks, route top score %.3f < %.3f — no match",
                route_top_score, self.settings.route_min_top_score,
            )
        elif not reranked_chunks:
            wide_fallback, wide_reason = True, "no chunk answered this on its own"
        elif matches_broad_keywords(question):
            wide_fallback, wide_reason = True, "question language implies broad coverage"
        else:
            # Assigned regardless of the verdict — the call (and its cost) happens
            # either way, unlike the two free/local checks above.
            self.bulk_llm.last_usage = None
            is_broad = self.intent_classifier.is_broad(question)
            _add_usage(tokens, self.bulk_llm.last_usage)
            if is_broad:
                wide_fallback, wide_reason = True, "classified as a broad-coverage question"
        timings["gate"] = (time.monotonic() - gate_started) * 1000

        if not wide_fallback:
            logger.info("  [5/6 compress] %d chunks in one batched call", len(reranked_chunks))
            self.llm.last_usage = None
            compress_started = time.monotonic()
            compressed_chunks = self.compressor.compress_batch(question, reranked_chunks)
            timings["compress"] = (time.monotonic() - compress_started) * 1000
            _add_usage(tokens, self.llm.last_usage)
            for chunk, compressed in zip(reranked_chunks, compressed_chunks):
                final_chunk_traces.append(
                    CompressedChunkTrace(
                        chunk=compressed if compressed is not None else chunk,
                        original_line_count=len(chunk.code.splitlines()),
                        compressed_line_count=(
                            len(compressed.code.splitlines()) if compressed else 0
                        ),
                        dropped=compressed is None,
                        original_tokens=len(chunk.code) // Tokenizer.CHARS_PER_TOKEN,
                        compressed_tokens=(
                            len(compressed.code) // Tokenizer.CHARS_PER_TOKEN if compressed else 0
                        ),
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
                    "  [5/6 compress] going wide (%s) — sending %d source(s), "
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
                        original_tokens=len(chunk.code) // Tokenizer.CHARS_PER_TOKEN,
                        compressed_tokens=len(chunk.code) // Tokenizer.CHARS_PER_TOKEN,
                    )
                    for chunk in wide_chunks
                ]
            elif wide_chunks:
                logger.info(
                    "  [5/6 compress] going wide (%s) but whole-source fallback (~%d "
                    "tokens) exceeds the %d token budget — refusing",
                    wide_reason, token_estimate, self.settings.wide_answer_max_tokens,
                )
                too_large_message = self._too_large_message(wide_file_count, token_estimate)
            else:
                logger.info(
                    "  [5/6 compress] going wide (%s) but no routed sources to fall back to",
                    wide_reason,
                )

        logger.info(
            "  [6/6 retrieval] %d chunks ready for generation (%.1fs elapsed)",
            len(final_chunks), time.monotonic() - started,
        )
        return _RetrievalState(
            query_vector=candidate.query_vector,
            routed=candidate.routed,
            scored_candidates=candidate.scored_candidates,
            reranked_scored=candidate.reranked_scored,
            final_chunks=final_chunks,
            final_chunk_traces=final_chunk_traces,
            wide_fallback=wide_fallback,
            wide_fallback_reason=wide_reason,
            too_large_message=too_large_message,
            diversity_picks=candidate.diversity_picks,
            sub_questions=sub_questions,
            routed_origin=routed_origin,
            candidate_origin=candidate_origin,
            reranked_origin=reranked_origin,
            timings=timings,
            tokens=tokens,
            tool_source_chunks=tool_source_chunks,
        )

    def _retrieve(
        self, question: str, space_id: str, decompose_result: DecomposeResult | None = None,
        decompose_ms: float = 0.0, decompose_tokens: dict[str, int] | None = None,
    ) -> _RetrievalState:
        """Runs decompose→embed→route→hybrid→rerank→compress — everything before
        generation. Split out so query_trace() (blocking) and query_trace_stream()
        (streaming) share one retrieval pass; only how the final answer text arrives
        differs between them.

        A compound question ("what does X do, and how is that different from Y?") is
        split into independent sub-questions first: a single embedding of a two-topic
        question is a blurry average of both, which hurts routing and hybrid search for
        either one. Each sub-question then runs stages 1-4 IN PARALLEL (bounded by
        max_subquestions), merges, and only then hits the shared compress/generate
        stages below. Most questions aren't compound — decompose() returns the question
        unchanged — so this is the same single pass as before for the common case.

        decompose_result lets a caller that already classified this turn (see
        Pipeline.route_question, called once per turn in chat_routes.py) pass that result
        straight through instead of paying for a second, redundant classification call —
        decompose() only runs here as a fallback for callers that didn't pre-route (the
        eval harness, direct pipeline tests). decompose_ms/decompose_tokens are that
        pre-routed call's own cost, so it still shows up in this trace's "decompose"
        timing/token totals instead of silently vanishing."""
        started = time.monotonic()
        logger.info("RETRIEVAL START | space=%s | question=%r", space_id, question)
        try:
            decompose_tokens = dict(decompose_tokens or {})
            if decompose_result is None:
                self.bulk_llm.last_usage = None
                decompose_started = time.monotonic()
                decompose_result = self.decomposer.decompose(question)
                decompose_ms = (time.monotonic() - decompose_started) * 1000
                _add_usage(decompose_tokens, self.bulk_llm.last_usage)
            mode = decompose_result.mode

            if mode == "single":
                sub_questions = decompose_result.sub_questions
                states = [
                    self._retrieve_candidates(
                        sub_questions[0], space_id, skip_hybrid_rerank=decompose_result.is_broad,
                        skip_rerank=decompose_result.wants_live_data or decompose_result.wants_report,
                    )
                ]
                decomposed: list[str] = []
            elif mode == "sequential":
                # Each hop's query doesn't exist until the previous hop's own retrieval
                # answers it — this chain cannot run in parallel like the independent case
                # below. Resolving a later hop's {hopN} placeholder costs one cheap
                # bulk-LLM call (QueryDecomposer.resolve_hop) over the previous hop's own
                # top-reranked chunk, skipped entirely when that hop found nothing. See
                # _hop_context.
                templates = decompose_result.sub_questions
                logger.info("  [0/6 decompose] sequential: hops=%r", templates)
                sub_questions = [templates[0]]
                states = [self._retrieve_candidates(templates[0], space_id)]
                for i, template in enumerate(templates[1:], start=2):
                    context = self._hop_context(states[-1])
                    placeholder = f"{{hop{i - 1}}}"
                    # Folded into the decompose stage's own timing/token bucket below
                    # (decompose_ms/decompose_tokens) rather than a separate stage — a
                    # resolve call is conceptually still "figuring out what to search
                    # for," same as the initial decompose call.
                    self.bulk_llm.last_usage = None
                    resolve_started = time.monotonic()
                    hop_q = self.decomposer.resolve_hop(sub_questions[-1], context, template, placeholder)
                    decompose_ms += (time.monotonic() - resolve_started) * 1000
                    _add_usage(decompose_tokens, self.bulk_llm.last_usage)
                    logger.info("  [hop %d/%d] resolved as %r", i, len(templates), hop_q)
                    sub_questions.append(hop_q)
                    states.append(self._retrieve_candidates(hop_q, space_id))
                decomposed = sub_questions
            else:
                sub_questions = decompose_result.sub_questions
                logger.info(
                    "  [0/6 decompose] split into %d sub-questions: %r",
                    len(sub_questions), sub_questions,
                )
                decomposed = sub_questions
                # Cancellation rides a ContextVar (src/cancellation.py) that FastAPI's
                # run_in_threadpool copies automatically; a bare ThreadPoolExecutor does
                # NOT do that for its workers, so without an explicit copy a browser
                # disconnect would silently fail to cancel these in-flight sub-retrievals.
                # A fresh copy_context() per task, not one shared copy: a Context can only
                # be entered (run()) by one call at a time, so reusing one across threads
                # raises "cannot enter context: ... already entered" the instant a second
                # thread tries to run() it concurrently.
                with ThreadPoolExecutor(max_workers=len(sub_questions)) as pool:
                    futures = [
                        pool.submit(contextvars.copy_context().run, self._retrieve_candidates, q, space_id)
                        for q in sub_questions
                    ]
                    states = [f.result() for f in futures]
            check()
            # Each sub-question's OWN rerank already gates on min_top_score (see
            # CrossReranker.rerank_scored) — an empty reranked_scored means THAT
            # sub-question specifically found nothing, before the merge below pools
            # everything together and that per-sub-question signal is lost. But a
            # non-empty result only proves relevance (it scored above the floor), not
            # answerability — a chunk can be squarely on-topic and still not contain the
            # specific fact asked for. Only worth the extra SufficiencyChecker call in the
            # ambiguous middle band (below sufficiency_check_max_score): a score
            # comfortably above it is empirically almost always a real, complete match
            # (weakest genuine match measured was 0.35 — see rerank_min_top_score's own
            # docstring), so spending a call to confirm near-certainty is wasted cost.
            # Applies regardless of mode now — a single, non-decomposed question gets the
            # same scrutiny a sub-question does, not just decomposed ones.
            insufficient_sub_qs: list[str] = []
            sufficiency_check_ms = 0.0
            sufficiency_check_tokens: dict[str, int] = {}
            # mode == "single" with is_broad is the one case reranked_scored is empty ON
            # PURPOSE (see skip_hybrid_rerank above) — the router already decided this
            # needs the wide-fallback path, not chunk-level search. Treating that empty
            # result as "insufficient, retry narrowly" would silently run a full search
            # the router deliberately skipped, and bypass wide-fallback entirely before
            # _finish_retrieval ever gets to make that call. Only skip the check for
            # THIS specific reason — a genuine zero-chunk result on any other path still
            # needs to be caught below. A live-data question is skipped for a different
            # reason: it isn't answered from reranked chunks at all (see
            # Pipeline._try_live_data_answer), so "insufficient, retry narrowly" would
            # burn a sufficiency-check call plus a full retry search — tens of seconds —
            # on a question neither step could ever satisfy, since there was never a
            # chunk-grounded answer to find here in the first place. Same reasoning
            # applies to a report question — see Pipeline._try_report_answer.
            skip_sufficiency_check = (
                (mode == "single" and decompose_result.is_broad)
                or decompose_result.wants_live_data or decompose_result.wants_report
            )
            if not skip_sufficiency_check:
                for q, s in zip(sub_questions, states):
                    if not s.reranked_scored:
                        insufficient_sub_qs.append(q)
                        continue
                    if s.reranked_scored[0][1] >= self.settings.sufficiency_check_max_score:
                        continue
                    self.bulk_llm.last_usage = None
                    check_started = time.monotonic()
                    missing = self.sufficiency_checker.check(
                        q, [c for c, _ in s.reranked_scored[: self.settings.rerank_top_k]]
                    )
                    sufficiency_check_ms += (time.monotonic() - check_started) * 1000
                    _add_usage(sufficiency_check_tokens, self.bulk_llm.last_usage)
                    if missing is not None:
                        insufficient_sub_qs.append(q)
            # One retry per insufficient sub-question, capped at a single extra pass (no
            # loop): rewrite it — the original phrasing may just not match how the source
            # text describes it, or may be missing the specific evidence type identified
            # above — and search again.
            retried_sub_questions: dict[str, str] = {}
            retry_ms = 0.0
            retry_tokens: dict[str, int] = {}
            if insufficient_sub_qs:
                # Snapshot before retry mutates states — a sub-question stays insufficient
                # unless retry actually replaces its state below. Re-deriving this from
                # `not s.reranked_scored` afterward would be wrong now that a sub-question
                # can be insufficient while still holding its ORIGINAL non-empty (just not
                # answering) chunks — a failed/no-op retry would then silently clear it.
                pre_retry_insufficient = set(insufficient_sub_qs)
                resolved: set[str] = set()
                retry_started = time.monotonic()

                def _retry_one(
                    idx: int, sub_q: str
                ) -> tuple[int, str, str, dict | None, _CandidateState | None]:
                    # last_usage is thread-local (see LLMClient) specifically so concurrent
                    # calls here don't race on a shared attribute — each thread reads back
                    # only its own call's usage.
                    self.bulk_llm.last_usage = None
                    rewritten = self.decomposer.rewrite_for_retry(sub_q, question)
                    usage = self.bulk_llm.last_usage
                    if rewritten == sub_q:
                        return idx, sub_q, rewritten, usage, None
                    logger.info("  [retry] %r found nothing, retrying as %r", sub_q, rewritten)
                    new_state = self._retrieve_candidates(rewritten, space_id)
                    return idx, sub_q, rewritten, usage, new_state

                # Independent per sub-question, same as the main parallel-retrieval pass
                # above — same ThreadPoolExecutor + copy_context() pattern, same reason
                # (cancellation propagation). Token totals are merged on this thread only,
                # after collecting every result, so retry_tokens (a plain shared dict)
                # never gets mutated from more than one thread at a time.
                targets = [(idx, sub_q) for idx, sub_q in enumerate(sub_questions) if sub_q in insufficient_sub_qs]
                with ThreadPoolExecutor(max_workers=len(targets)) as pool:
                    futures = [
                        pool.submit(contextvars.copy_context().run, _retry_one, idx, sub_q)
                        for idx, sub_q in targets
                    ]
                    for f in futures:
                        idx, sub_q, rewritten, usage, new_state = f.result()
                        _add_usage(retry_tokens, usage)
                        if new_state is None:
                            continue
                        retried_sub_questions[sub_q] = rewritten
                        if new_state.reranked_scored:
                            states[idx] = new_state
                            resolved.add(sub_q)
                retry_ms = (time.monotonic() - retry_started) * 1000
                insufficient_sub_qs = [q for q in pre_retry_insufficient if q not in resolved]

            if not insufficient_sub_qs:
                sufficiency = "sufficient"
            elif len(insufficient_sub_qs) == len(states):
                sufficiency = "insufficient"
            else:
                sufficiency = "partial"

            candidate, routed_origin, candidate_origin, reranked_origin = self._merge_candidates(
                sub_questions, states
            )
            state = self._finish_retrieval(
                question, space_id, candidate, decomposed,
                routed_origin, candidate_origin, reranked_origin, started,
                wants_live_data=decompose_result.wants_live_data, wants_report=decompose_result.wants_report,
            )
            state.timings["decompose"] = decompose_ms
            _add_usage(state.tokens, decompose_tokens)
            if sufficiency_check_ms:
                state.timings["sufficiency_check"] = sufficiency_check_ms
                _add_usage(state.tokens, sufficiency_check_tokens)
            if retried_sub_questions:
                state.timings["retry"] = retry_ms
                _add_usage(state.tokens, retry_tokens)
            state.sufficiency = sufficiency
            state.insufficient_sub_questions = insufficient_sub_qs
            state.retried_sub_questions = retried_sub_questions
            state.decompose_mode = mode
            state.is_broad = decompose_result.is_broad
            state.wants_chart = decompose_result.wants_chart
            state.wants_live_data = decompose_result.wants_live_data
            state.wants_report = decompose_result.wants_report
        except Cancelled:
            logger.warning(
                "RETRIEVAL CANCELLED after %.1fs | question=%r", time.monotonic() - started, question
            )
            raise
        return state

    def _build_query_trace(
        self, question: str, space_id: str, state: _RetrievalState, answer_trace: AnswerTrace,
        raw_question: str | None, history: list[tuple[str, str]] | None,
        live_data_tool: LiveDataToolTrace | None = None, report_tool: ReportToolTrace | None = None,
    ) -> QueryTrace:
        return QueryTrace(
            question=question,
            space_id=space_id,
            query_embedding=vector_preview(state.query_vector),
            sub_questions=state.sub_questions,
            routed_files=[
                RoutedFile(file_path=fp, score=score, source_question=state.routed_origin.get(fp))
                for fp, _source_id, score in state.routed
            ],
            candidates=[
                ScoredChunkTrace(
                    chunk=sc.chunk,
                    dense_score=sc.dense_score,
                    bm25_score=sc.bm25_score,
                    fused_score=sc.fused_score,
                    source_question=state.candidate_origin.get(sc.chunk.id),
                )
                for sc in state.scored_candidates
            ],
            reranked=[
                RerankedChunkTrace(
                    chunk=chunk, rerank_score=score,
                    source_question=state.reranked_origin.get(chunk.id),
                    diversity_pick=chunk.id in state.diversity_picks,
                )
                for chunk, score in state.reranked_scored
            ],
            final_chunks=state.final_chunk_traces,
            rerank_min_top_score=self.settings.rerank_min_top_score,
            wide_fallback=state.wide_fallback,
            wide_fallback_reason=state.wide_fallback_reason,
            system_prompt=(
                LIVE_DATA_SYSTEM_PROMPT if live_data_tool
                else "(no LLM call — see below)" if report_tool
                else SYSTEM_PROMPT
            ),
            final_prompt=(
                f"{live_data_tool.context}\n\nQuestion: {question}" if live_data_tool
                else (
                    f"{report_tool.context}\n\n"
                    "The table/chart/text above is built directly from this fetched data in "
                    "code, not generated by an LLM call — see Pipeline._try_report_answer / "
                    "build_report_block."
                ) if report_tool
                else self.answer_generator.build_prompt(
                    raw_question or question, state.final_chunks, state.insufficient_sub_questions
                )
            ),
            history=[{"role": role, "content": content} for role, content in (history or [])],
            answer=answer_trace,
            timings=state.timings,
            tokens=state.tokens,
            sufficiency=state.sufficiency,
            insufficient_sub_questions=state.insufficient_sub_questions,
            retried_sub_questions=state.retried_sub_questions,
            decompose_mode=state.decompose_mode,
            is_broad=state.is_broad,
            wants_chart=state.wants_chart,
            live_data_tool=live_data_tool,
            report_tool=report_tool,
        )

    def query_trace(
        self, question: str, space_id: str,
        raw_question: str | None = None, history: list[tuple[str, str]] | None = None,
        decompose_result: DecomposeResult | None = None,
        decompose_ms: float = 0.0, decompose_tokens: dict[str, int] | None = None,
    ) -> QueryTrace:
        """Runs the same retrieval stages as query(), returning every stage's intermediate
        scores/chunks alongside the final answer."""
        started = time.monotonic()
        state = self._retrieve(question, space_id, decompose_result, decompose_ms, decompose_tokens)
        self.llm.last_usage = None
        # Reset here too, not just before decompose/retry — _answer_trace's claim
        # attribution call (see ClaimAttributor) is the last bulk_llm call in this
        # request, and without this reset a stale value from an earlier stage would get
        # double-counted into state.tokens below on any path that skips attribution.
        self.bulk_llm.last_usage = None
        generate_started = time.monotonic()
        # `question` (the rewritten standalone form), never raw_question — unlike the
        # normal chunk-grounded answer call below, which deliberately prefers
        # raw_question for natural phrasing, this question is used to MATCH a place
        # by name (see find_matching_places). raw_question can still hold an
        # unresolved pronoun ("...there") that the rewrite step exists specifically
        # to resolve; matching against it defeats that resolution entirely.
        live_result = (
            self._try_live_data_answer(question, space_id, self._live_data_candidate_chunks(state))
            if state.wants_live_data else None
        )
        report_result = (
            self._try_report_answer(question, space_id, self._live_data_candidate_chunks(state), wants_chart=state.wants_chart)
            if state.wants_report and live_result is None else None
        )
        live_tool_trace = None
        report_tool_trace = None
        if live_result is not None:
            live_text, live_table, live_tool_trace, live_citations = live_result
            answer_trace = AnswerTrace(
                text=live_text, model=self.settings.llm_model, live_data=True, table=live_table,
                citations=live_citations,
            )
        elif report_result is not None:
            report_text, report_table, report_chart, report_tool_trace = report_result
            answer_trace = AnswerTrace(
                text=report_text, model=self.settings.llm_model, table=report_table, chart=report_chart
            )
        else:
            # Live/report classification fired but no place/device matched (or, for
            # report, the time window couldn't be resolved) — final_chunks is empty
            # because wants_live_data/wants_report skipped rerank and wide-fallback, not
            # because nothing relevant was found. Fall back to the raw hybrid candidates
            # instead of a false NO_MATCH (see _try_live_data_answer's docstring).
            if (state.wants_live_data or state.wants_report) and not state.final_chunks:
                state.final_chunks = self._live_data_candidate_chunks(state)
            answer_trace = self._answer_trace(
                raw_question or question, state.final_chunks, history,
                empty_message=state.too_large_message, insufficient=state.insufficient_sub_questions,
                wants_chart=state.wants_chart,
            )
        tool_answered = live_result is not None or report_result is not None
        if state.final_chunks or tool_answered:
            state.timings["generate"] = (time.monotonic() - generate_started) * 1000
            _add_usage(state.tokens, self.llm.last_usage)
            _add_usage(state.tokens, self.bulk_llm.last_usage)
        logger.info("RETRIEVAL DONE in %.1fs", time.monotonic() - started)
        return self._build_query_trace(
            question, space_id, state, answer_trace, raw_question, history,
            live_data_tool=live_tool_trace, report_tool=report_tool_trace,
        )

    def query_trace_stream(
        self, question: str, space_id: str,
        raw_question: str | None = None, history: list[tuple[str, str]] | None = None,
        decompose_result: DecomposeResult | None = None,
        decompose_ms: float = 0.0, decompose_tokens: dict[str, int] | None = None,
    ) -> Generator[str, None, QueryTrace]:
        """Same as query_trace(), but yields the answer's text deltas as they arrive
        instead of blocking for the full response. Once exhausted, the completed
        QueryTrace is available as the generator's return value (StopIteration.value) —
        built only once the full text (and its chart, if any) is known."""
        started = time.monotonic()
        state = self._retrieve(question, space_id, decompose_result, decompose_ms, decompose_tokens)

        # Tried before the final_chunks check, and against the wider candidate set (see
        # _live_data_candidate_chunks) — the wide-fallback gate is skipped entirely for a
        # live-data question (see _finish_retrieval's wants_live_data), so final_chunks
        # can legitimately be empty here even when the right chunk exists among the raw
        # hybrid-search candidates. Not token-streamed when it fires — one blocking LLM
        # call over a short live-data context, same non-streaming precedent as
        # answer_meta(); the consumer (chat_routes._run_turn) treats any yielded string
        # as a delta either way, so a single whole-text yield is a transparent
        # substitution.
        live_result = (
            # `question` (the rewritten standalone form), never raw_question — unlike the
            # normal chunk-grounded answer call below, which deliberately prefers
            # raw_question for natural phrasing, this question is used to MATCH a place
            # by name (see find_matching_places). raw_question can still hold an
            # unresolved pronoun ("...there") that the rewrite step exists specifically
            # to resolve; matching against it defeats that resolution entirely.
            self._try_live_data_answer(question, space_id, self._live_data_candidate_chunks(state))
            if state.wants_live_data else None
        )
        report_result = (
            self._try_report_answer(question, space_id, self._live_data_candidate_chunks(state), wants_chart=state.wants_chart)
            if state.wants_report and live_result is None else None
        )
        live_tool_trace = None
        report_tool_trace = None
        if live_result is not None:
            live_text, live_table, live_tool_trace, live_citations = live_result
            generate_started = time.monotonic()
            yield live_text
            answer_trace = AnswerTrace(
                text=live_text, model=self.settings.llm_model, live_data=True, table=live_table,
                citations=live_citations,
            )
            state.timings["generate"] = (time.monotonic() - generate_started) * 1000
            _add_usage(state.tokens, self.llm.last_usage)
            _add_usage(state.tokens, self.bulk_llm.last_usage)
        elif report_result is not None:
            report_text, report_table, report_chart, report_tool_trace = report_result
            generate_started = time.monotonic()
            yield report_text
            answer_trace = AnswerTrace(
                text=report_text, model=self.settings.llm_model, table=report_table, chart=report_chart
            )
            state.timings["generate"] = (time.monotonic() - generate_started) * 1000
            _add_usage(state.tokens, self.llm.last_usage)
            _add_usage(state.tokens, self.bulk_llm.last_usage)
        else:
            # Live/report classification fired but no place/device matched (or, for
            # report, the time window couldn't be resolved) — final_chunks is empty
            # because wants_live_data/wants_report skipped rerank and wide-fallback, not
            # because nothing relevant was found. Fall back to the raw hybrid candidates
            # instead of a false NO_MATCH (see _try_live_data_answer's docstring, and
            # query_trace()'s matching fallback).
            if (state.wants_live_data or state.wants_report) and not state.final_chunks:
                state.final_chunks = self._live_data_candidate_chunks(state)
            if not state.final_chunks:
                text = state.too_large_message or self.NO_MATCH
                yield text
                answer_trace = AnswerTrace(text=text, model=self.settings.llm_model)
                logger.info("RETRIEVAL DONE in %.1fs", time.monotonic() - started)
                return self._build_query_trace(question, space_id, state, answer_trace, raw_question, history)
            check()  # live/report tool traces stay None on this fallback path — no tool call to show
            self.llm.last_usage = None
            # See query_trace()'s matching reset — without it a stale bulk_llm usage
            # from an earlier stage would double-count into state.tokens below on the
            # RuntimeError branch, which never reaches the attribution call.
            self.bulk_llm.last_usage = None
            generate_started = time.monotonic()
            full_text = ""
            try:
                for delta in self.answer_generator.answer_stream(
                    raw_question or question, state.final_chunks, history=history,
                    insufficient=state.insufficient_sub_questions, wants_chart=state.wants_chart,
                ):
                    full_text += delta
                    yield delta
                answer = self.answer_generator.finalize(full_text)
                citations = self.claim_attributor.attribute(answer.text, state.final_chunks)
                answer_trace = AnswerTrace(
                    text=answer.text, model=self.settings.llm_model, chart=answer.chart,
                    citations=citations,
                )
            except RuntimeError as exc:
                # Persist whatever streamed before the failure — that's what the user
                # already saw — alongside the error, rather than discarding it.
                answer_trace = AnswerTrace(
                    text=full_text, model=self.settings.llm_model, error=str(exc)
                )
            state.timings["generate"] = (time.monotonic() - generate_started) * 1000
            _add_usage(state.tokens, self.llm.last_usage)
            _add_usage(state.tokens, self.bulk_llm.last_usage)

        logger.info("RETRIEVAL DONE in %.1fs", time.monotonic() - started)
        return self._build_query_trace(
            question, space_id, state, answer_trace, raw_question, history,
            live_data_tool=live_tool_trace, report_tool=report_tool_trace,
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
