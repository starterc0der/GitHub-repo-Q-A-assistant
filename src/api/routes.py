from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import settings
from src.ingest_cache import IngestCache
from src.pipeline import Pipeline
from src.trace import IngestProgress

logger = logging.getLogger(__name__)

router = APIRouter()
pipeline = Pipeline(settings)
ingest_cache = IngestCache(settings.ingest_cache_dir)


class IngestRequest(BaseModel):
    url: str


class QueryRequest(BaseModel):
    question: str
    repo: str


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    report = pipeline.ingest_repo(request.url)
    return {
        "repo": report.repo,
        "file_count": report.file_count,
        "chunk_count": report.chunk_count,
    }


@router.post("/query")
def query(request: QueryRequest) -> dict[str, object]:
    answer = pipeline.query(request.question, request.repo)
    return {
        "text": answer.text,
        "citations": [
            {
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "snippet": c.snippet,
            }
            for c in answer.citations
        ],
        "confidence": answer.confidence,
    }


@router.post("/ingest/trace")
def ingest_trace(request: IngestRequest) -> dict[str, object]:
    """Same ingest as /ingest, but returns per-file/per-chunk detail instead of counts —
    powers the pipeline-visualization UI. Cached per repo: the chunk/summary vectors
    already persist in Qdrant after a first successful ingest, so a cache hit here skips
    clone/summarize/embed entirely and just replays the saved trace."""
    repo = Pipeline.repo_name(request.url)
    cached = ingest_cache.load(repo)
    if cached is not None:
        return cached

    result = asdict(pipeline.ingest_repo_trace(request.url))
    ingest_cache.save(repo, result)
    return result


@router.get("/repos")
def list_repos() -> dict[str, object]:
    """Every repo with a cached ingest trace — powers the repo-picker dropdown."""
    return {"repos": ingest_cache.list_repos()}


@router.get("/ingest/trace/stream")
def ingest_trace_stream(url: str) -> StreamingResponse:
    """Same ingest as POST /ingest/trace, but streams progress via Server-Sent Events —
    GET rather than POST because the browser's EventSource can't send a request body.
    Powers the ingestion progress bar, which matters most on large repos where the
    per-file summarize/contextualize LLM calls can otherwise run for minutes with no
    other feedback."""

    def events() -> Iterator[str]:
        repo = Pipeline.repo_name(url)
        cached = ingest_cache.load(repo)
        if cached is not None:
            yield f"data: {json.dumps({'type': 'complete', 'trace': cached})}\n\n"
            return

        try:
            for event in pipeline.ingest_repo_trace_stream(url):
                if isinstance(event, IngestProgress):
                    yield f"data: {json.dumps({'type': 'progress', **asdict(event)})}\n\n"
                else:
                    result = asdict(event)
                    ingest_cache.save(repo, result)
                    yield f"data: {json.dumps({'type': 'complete', 'trace': result})}\n\n"
        except Exception as exc:
            # Broad on purpose: once streaming has started the response is already 200
            # OK, so any failure has to become a well-formed SSE error event instead of
            # silently killing the connection.
            logger.exception("Ingest stream failed for %s", url)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/query/trace")
def query_trace(request: QueryRequest) -> dict[str, object]:
    """Same retrieval as /query, but stops before the answer LLM call and returns every
    stage's intermediate chunks/scores — powers the pipeline-visualization UI."""
    return asdict(pipeline.query_trace(request.question, request.repo))
