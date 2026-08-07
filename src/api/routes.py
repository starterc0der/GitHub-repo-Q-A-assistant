from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client.http.exceptions import UnexpectedResponse

from src.cancellation import Cancelled, reset_canceller, set_canceller
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


@router.get("/repos/{repo}/trace")
def cached_ingest_trace(repo: str) -> dict[str, object]:
    """Replay a previously-ingested repo's trace by name, for the ingest-tab picker.

    Deliberately not under /ingest/trace/{repo}: that would shadow the literal
    /ingest/trace/stream route below it. Cache-only — never re-ingests, so selecting a
    repo here costs nothing and can't fire LLM calls.
    """
    cached = ingest_cache.load(repo)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"No cached ingest for {repo!r}")
    return cached


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


async def _run_cancellable(request: Request, work: Callable[[], object]) -> object:
    """Run blocking pipeline work in a threadpool, aborting it if the client goes away.

    Cooperative, since threads can't be killed: work stops at its next checkpoint, which
    frees the rate-limit budget instead of holding it for a full retry backoff.
    """
    cancelled = threading.Event()

    async def watch_client() -> None:
        while not cancelled.is_set():
            if await request.is_disconnected():
                cancelled.set()
                return
            await asyncio.sleep(0.4)

    def runner() -> object:
        token = set_canceller(cancelled.is_set)
        try:
            return work()
        finally:
            reset_canceller(token)

    watcher = asyncio.create_task(watch_client())
    try:
        return await run_in_threadpool(runner)
    finally:
        cancelled.set()  # stop the poller whichever way we exited
        watcher.cancel()


@router.post("/query/trace")
async def query_trace(request: Request, body: QueryRequest) -> Response:
    """Same retrieval as /query, returning every stage's intermediate chunks/scores —
    powers the pipeline-visualization UI.

    Cancels itself if the browser navigates away mid-query, so a refresh doesn't leave
    LLM calls running against a rate limit the next query needs.
    """
    try:
        result = await _run_cancellable(
            request, lambda: asdict(pipeline.query_trace(body.question, body.repo))
        )
    except UnexpectedResponse as exc:
        # Trace cache and vector store can disagree — e.g. `docker compose down -v` on one
        # volume but not the other. Say so instead of surfacing a raw Qdrant 404.
        if exc.status_code != 404:
            raise
        raise HTTPException(
            status_code=409,
            detail=f"{body.repo!r} has a cached trace but no vectors in Qdrant. Re-ingest it.",
        ) from exc
    except Cancelled:
        # 499 is nginx's "client closed request" — nothing is listening, but this keeps
        # the log honest instead of reporting a 500 for a request we abandoned on purpose.
        logger.info("Query cancelled by client disconnect: %r", body.question)
        return Response(status_code=499)
    return JSONResponse(result)
