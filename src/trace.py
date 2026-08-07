from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
    repo: str
    repo_url: str
    repo_summary: str
    clone: CloneTrace
    walk: WalkTrace
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


@dataclass
class ScoredChunkTrace:
    chunk: CodeChunk
    dense_score: float
    bm25_score: float
    fused_score: float


@dataclass
class RerankedChunkTrace:
    chunk: CodeChunk
    rerank_score: float


@dataclass
class CompressedChunkTrace:
    chunk: CodeChunk
    original_line_count: int
    compressed_line_count: int
    dropped: bool


@dataclass
class CitationTrace:
    file_path: str
    start_line: int
    end_line: int
    snippet: str


@dataclass
class AnswerTrace:
    """The final generation step. Unlike every other stage this one actually calls the
    answer LLM, so a query trace costs one Opus request on top of the compression calls."""

    text: str
    citations: list[CitationTrace] = field(default_factory=list)
    confidence: float = 1.0
    model: str = ""
    error: str = ""


@dataclass
class QueryTrace:
    question: str
    repo: str
    query_embedding: EmbeddingPreview
    routed_files: list[RoutedFile] = field(default_factory=list)
    candidates: list[ScoredChunkTrace] = field(default_factory=list)
    reranked: list[RerankedChunkTrace] = field(default_factory=list)
    final_chunks: list[CompressedChunkTrace] = field(default_factory=list)
    system_prompt: str = ""
    final_prompt: str = ""
    answer: AnswerTrace | None = None
    # Surfaced so the UI can say *why* a stage is empty rather than showing a blank list.
    rerank_min_top_score: float = 0.0
    # Two PCA spaces for the vector-space visualization: file-summary embeddings (all
    # files in the repo, for the routing stages) and whole-repo chunk embeddings (every
    # chunk, not just the routed-file pool — shared with the hybrid/rerank/compress plots
    # and the "whole vector space" modal, so the query lands in the same spot everywhere
    # a chunk-level plot is shown). The query is projected separately into each space, so
    # its position differs between the file-summary view and the chunk-level views.
    query_file_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    file_xyz: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # "{file_path} · {symbol_name or 'block'}" per chunk id — covers every chunk in the
    # repo (see whole_chunk_xyz below), not just the routed-file pool, so the vector-space
    # hover text never falls back to guessing from the id string.
    chunk_labels: dict[str, str] = field(default_factory=dict)
    query_whole_chunk_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    whole_chunk_xyz: dict[str, tuple[float, float, float]] = field(default_factory=dict)
