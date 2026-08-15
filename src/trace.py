from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.generate.provenance import ClaimCitation
from src.index.schema import CodeChunk
from src.ingest.repo_loader import SkippedFile


@dataclass
class EmbeddingPreview:
    """A human-inspectable stand-in for a full embedding vector (too many dims to show raw)."""

    dim: int
    norm: float
    preview: list[float]


def vector_preview(vector: list[float], n: int = 8) -> EmbeddingPreview:
    return EmbeddingPreview(
        dim=len(vector),
        norm=round(float(np.linalg.norm(vector)), 4),
        preview=[round(v, 4) for v in vector[:n]],
    )


DIMS = 3


def project_3d(vectors: list[list[float]]) -> list[tuple[float, float, float]]:
    """PCA projection to 3D via SVD — a real dimensionality reduction of the actual
    embeddings passed in, not a hand-placed layout. Conveys relative distance only;
    axes/scale carry no independent meaning. Degenerates to the origin for <2 vectors,
    since PCA needs at least 2 points to define a spread.

    Three components rather than two because PC3 is not vestigial on these embeddings —
    measured at ~82% of PC2's variance on a real ingest, taking the visible share of total
    variance from ~15% to ~21%. Points that overlap in 2D are often genuinely apart here.
    """
    if len(vectors) < 2:
        return [(0.0, 0.0, 0.0) for _ in vectors]
    matrix = np.array(vectors, dtype=float)
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(DIMS, vt.shape[0])
    coords = centered @ vt[:k].T
    if k < DIMS:
        coords = np.pad(coords, ((0, 0), (0, DIMS - k)))
    return [(round(float(x), 4), round(float(y), 4), round(float(z), 4)) for x, y, z in coords]


@dataclass
class ChunkTrace:
    chunk: CodeChunk
    embedding: EmbeddingPreview
    xyz: tuple[float, float, float]


@dataclass
class FileTrace:
    file_path: str
    language: str
    summary: str
    symbols: list[str]
    summary_embedding: EmbeddingPreview
    chunks: list[ChunkTrace] = field(default_factory=list)


@dataclass
class CloneTrace:
    depth: int
    commit: str
    local_path: str


@dataclass
class WalkTrace:
    total_scanned: int
    kept: int
    skipped: list[SkippedFile] = field(default_factory=list)


@dataclass
class IngestTrace:
    space_id: str
    source_id: str
    name: str
    kind: str
    uri: str | None = None
    summary: str = ""
    clone: CloneTrace | None = None
    walk: WalkTrace | None = None
    files: list[FileTrace] = field(default_factory=list)


@dataclass
class IngestProgress:
    """One update from the ingest generator — powers the progress bar on large repos,
    where the per-file summarize/contextualize LLM calls can take a while with no other
    feedback."""

    stage: str  # "clone" | "walk" | "summarize_repo" | "process_files" | "embed"
    message: str
    files_done: int = 0
    files_total: int = 0


@dataclass
class RoutedFile:
    file_path: str
    score: float
    # Which sub-question produced this row, when the question was decomposed — None on
    # the (common) undecomposed path. See QueryTrace.sub_questions.
    source_question: str | None = None


@dataclass
class ScoredChunkTrace:
    chunk: CodeChunk
    dense_score: float
    bm25_score: float
    fused_score: float
    source_question: str | None = None


@dataclass
class RerankedChunkTrace:
    chunk: CodeChunk
    rerank_score: float
    source_question: str | None = None


@dataclass
class CompressedChunkTrace:
    chunk: CodeChunk
    original_line_count: int
    compressed_line_count: int
    dropped: bool


@dataclass
class AnswerTrace:
    """The final generation step. Unlike every other stage this one actually calls the
    answer LLM, so a query trace costs one main-model call on top of the compression calls."""

    text: str
    model: str = ""
    error: str = ""
    # {"title", "categories": [...], "series": [{"name", "values": [...]}]} — only set on
    # comparison/graph questions where the model emitted a chart block; see ChartParser.
    chart: dict | None = None
    # Per-claim provenance computed after generation, not self-reported by the answer
    # model — see ClaimAttributor. Empty when there were no chunks/answer to attribute
    # (NO_MATCH, gated, or a failed generation). The chat UI never renders these; they
    # exist only for the pipeline trace view.
    citations: list[ClaimCitation] = field(default_factory=list)


@dataclass
class QueryTrace:
    question: str
    space_id: str
    query_embedding: EmbeddingPreview
    # Populated only when `question` was split into independent sub-questions; empty on
    # the ordinary single-pass path. See RoutedFile/ScoredChunkTrace/RerankedChunkTrace's
    # source_question to see which sub-question produced which row below.
    sub_questions: list[str] = field(default_factory=list)
    routed_files: list[RoutedFile] = field(default_factory=list)
    candidates: list[ScoredChunkTrace] = field(default_factory=list)
    reranked: list[RerankedChunkTrace] = field(default_factory=list)
    final_chunks: list[CompressedChunkTrace] = field(default_factory=list)
    system_prompt: str = ""
    final_prompt: str = ""
    # The generate call actually receives system_prompt, THEN these turns, THEN
    # final_prompt as the real chat messages (see LLMClient.complete) — not folded into
    # final_prompt's text. Surfaced separately so "final prompt, verbatim" is actually
    # verbatim. Empty on turn 1, where there's no history to send.
    history: list[dict] = field(default_factory=list)
    answer: AnswerTrace | None = None
    # Surfaced so the UI can say *why* a stage is empty rather than showing a blank list.
    rerank_min_top_score: float = 0.0
    # Whole source(s) sent instead of the rerank-scored top-k; reason is the human-readable why.
    wide_fallback: bool = False
    wide_fallback_reason: str = ""
    # Stage key -> milliseconds. Keys: cache, decompose, embed, route, hybrid, rerank,
    # gate, compress, generate. compress/generate are absent (not 0) when skipped
    # entirely (wide fallback, or a gated NO_MATCH) — every other stage always runs at
    # least its cheap local check, so it always has a real (if small) number.
    timings: dict[str, float] = field(default_factory=dict)
    # Total prompt/completion tokens across every LLM call this query made (decompose +
    # compress + generate, plus broad-intent when it fires). Absent, not zero, when no
    # provider in the chain reported usage — see LLMClient.last_usage.
    tokens: dict[str, int] = field(default_factory=dict)
    # Only meaningful when the question was decomposed: "sufficient" (every sub-question
    # found evidence), "partial" (some did, some didn't — see insufficient_sub_questions),
    # "insufficient" (none did — same case the existing NO_MATCH/wide_fallback gate
    # already catches). A single, undecomposed question is always "sufficient" here; its
    # own coverage is already fully captured by wide_fallback/NO_MATCH.
    sufficiency: str = "sufficient"
    insufficient_sub_questions: list[str] = field(default_factory=list)
    # original sub-question -> its retry rewrite, for every insufficient sub-question that
    # got one retry attempt before sufficiency/insufficient_sub_questions above were
    # finalized. Empty unless at least one sub-question needed a retry. See Pipeline._retrieve.
    retried_sub_questions: dict[str, str] = field(default_factory=dict)
    # "single" | "parallel" | "sequential" — see QueryDecomposer.decompose. Only changes
    # how sub_questions should be captioned/labeled in the UI (independent parts vs.
    # chained hops); every stage after decompose treats sub_questions identically.
    decompose_mode: str = "single"
    # From the same upfront classification (see DecomposeResult) — is_broad means hybrid
    # search + rerank were skipped entirely in favor of the wide-source path;
    # wants_chart means the answer prompt got an explicit chart-formatting hint.
    is_broad: bool = False
    wants_chart: bool = False


@dataclass
class VectorsTrace:
    """The vector-space visualization data — split out from QueryTrace because it's a
    full-space PCA over every file/chunk embedding, expensive relative to the rest of a
    query, and only needed if the user opens the pipeline-breakdown UI. Computed on
    demand by Pipeline.vectors_trace() instead of eagerly on every message send.

    Two PCA spaces: file-summary embeddings (all files in the repo, for the routing
    stages) and whole-repo chunk embeddings (every chunk, not just the routed-file pool —
    shared with the hybrid/rerank/compress plots and the "whole vector space" modal, so
    the query lands in the same spot everywhere a chunk-level plot is shown). The query is
    projected separately into each space, so its position differs between the two.
    """

    query_file_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    file_xyz: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # "{file_path} · {symbol_name or 'block'}" per chunk id — covers every chunk in the
    # repo (see whole_chunk_xyz below), not just the routed-file pool, so the vector-space
    # hover text never falls back to guessing from the id string.
    chunk_labels: dict[str, str] = field(default_factory=dict)
    query_whole_chunk_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    whole_chunk_xyz: dict[str, tuple[float, float, float]] = field(default_factory=dict)
