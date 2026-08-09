from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import settings
from src.db import connect, init_db, new_id, now, sweep_stale_ingests
from src.pipeline import Pipeline
from src.trace import IngestProgress, IngestTrace

logger = logging.getLogger(__name__)

router = APIRouter()
pipeline = Pipeline(settings)

init_db(settings.db_path)
_swept = sweep_stale_ingests(settings.db_path)
if _swept:
    logger.warning("Marked %d interrupted ingest(s) as failed on startup", _swept)

# One lock per space serializes concurrent ingests into it — Pipeline is a shared
# singleton, so two background ingests into the same space would otherwise race.
_space_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(space_id: str) -> threading.Lock:
    with _locks_guard:
        return _space_locks.setdefault(space_id, threading.Lock())


# --------------------------------------------------------------------- spaces


class CreateSpaceRequest(BaseModel):
    name: str
    description: str = ""
    color: str = "accent"


class UpdateSpaceRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None
    rerank_min_top_score: float | None = None


def _space_row(space_id: str) -> dict[str, object]:
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM spaces WHERE id=?", (space_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No space {space_id!r}")
    return dict(row)


@router.post("/spaces")
def create_space(request: CreateSpaceRequest) -> dict[str, object]:
    space_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, description, color, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (space_id, request.name, request.description, request.color, now(), now()),
        )
    return _space_row(space_id)


@router.get("/spaces")
def list_spaces() -> dict[str, object]:
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT s.*, COUNT(src.id) AS source_count FROM spaces s "
            "LEFT JOIN sources src ON src.space_id = s.id "
            "GROUP BY s.id ORDER BY s.updated_at DESC"
        ).fetchall()
    return {"spaces": [dict(r) for r in rows]}


@router.get("/spaces/{space_id}")
def get_space(space_id: str) -> dict[str, object]:
    space = _space_row(space_id)
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM sources WHERE space_id=? ORDER BY created_at", (space_id,)
        ).fetchall()
    return {**space, "sources": [_source_dict(r) for r in rows]}


@router.patch("/spaces/{space_id}")
def update_space(space_id: str, request: UpdateSpaceRequest) -> dict[str, object]:
    _space_row(space_id)
    fields = request.model_dump(exclude_unset=True)
    if fields:
        set_clause = ", ".join(f"{k}=?" for k in fields)
        with connect(settings.db_path) as conn:
            conn.execute(
                f"UPDATE spaces SET {set_clause}, updated_at=? WHERE id=?",
                (*fields.values(), now(), space_id),
            )
    return _space_row(space_id)


@router.delete("/spaces/{space_id}")
def delete_space(space_id: str) -> dict[str, str]:
    _space_row(space_id)
    pipeline.delete_space(space_id)
    with connect(settings.db_path) as conn:
        conn.execute("DELETE FROM spaces WHERE id=?", (space_id,))  # cascades sources/chats/messages
    return {"status": "deleted"}


# -------------------------------------------------------------------- sources


class CreateSourceRequest(BaseModel):
    kind: Literal["repo", "text"]
    name: str | None = None
    uri: str | None = None  # repo clone URL
    text: str | None = None  # pasted text


def _source_dict(row: object) -> dict[str, object]:
    d = dict(row)
    d["has_trace"] = d.pop("ingest_trace") is not None
    d["meta"] = json.loads(d["meta"]) if d["meta"] else {}
    return d


def _get_source_row(source_id: str) -> dict[str, object]:
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No source {source_id!r}")
    return dict(row)


def _create_source_row(space_id: str, kind: str, name: str, uri: str | None) -> str:
    source_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO sources (id, space_id, kind, name, uri, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, space_id, kind, name, uri, now()),
        )
    return source_id


def _run_ingest_background(
    space_id: str, source_id: str, kind: str, name: str, uri: str | None, text: str | None
) -> None:
    with _lock_for(space_id):
        with connect(settings.db_path) as conn:
            conn.execute("UPDATE sources SET status='ingesting' WHERE id=?", (source_id,))
        try:
            trace: IngestTrace | None = None
            for event in pipeline.ingest_source_trace_stream(space_id, source_id, kind, name, uri, text):
                if isinstance(event, IngestProgress):
                    with connect(settings.db_path) as conn:
                        conn.execute(
                            "UPDATE sources SET meta=? WHERE id=?",
                            (json.dumps({"progress": asdict(event)}), source_id),
                        )
                else:
                    trace = event
        except Exception as exc:
            logger.exception("Ingest failed for source %s (%s)", source_id, name)
            with connect(settings.db_path) as conn:
                conn.execute(
                    "UPDATE sources SET status='failed', error=?, meta='{}' WHERE id=?",
                    (str(exc), source_id),
                )
            return

        file_count = len(trace.files)
        chunk_count = sum(len(f.chunks) for f in trace.files)
        with connect(settings.db_path) as conn:
            conn.execute(
                "UPDATE sources SET status='ready', file_count=?, chunk_count=?, "
                "ingest_trace=?, meta='{}', ingested_at=? WHERE id=?",
                (file_count, chunk_count, json.dumps(asdict(trace)), now(), source_id),
            )
            # A source finishing ingest can change the answer to any question in this
            # space — invalidate wholesale rather than guess which cached answers still hold.
            conn.execute("DELETE FROM qa_cache WHERE space_id=?", (space_id,))


@router.get("/spaces/{space_id}/sources")
def list_sources(space_id: str) -> dict[str, object]:
    _space_row(space_id)
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM sources WHERE space_id=? ORDER BY created_at", (space_id,)
        ).fetchall()
    return {"sources": [_source_dict(r) for r in rows]}


@router.post("/spaces/{space_id}/sources")
def create_source(space_id: str, request: CreateSourceRequest) -> dict[str, object]:
    _space_row(space_id)
    if request.kind == "repo":
        if not request.uri:
            raise HTTPException(status_code=422, detail="repo sources require a uri")
        name = request.name or Pipeline.default_source_name(request.uri)
    else:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=422, detail="text sources require non-empty text")
        name = request.name or "Pasted text"

    source_id = _create_source_row(space_id, request.kind, name, request.uri)
    threading.Thread(
        target=_run_ingest_background,
        args=(space_id, source_id, request.kind, name, request.uri, request.text),
        daemon=True,
    ).start()
    return _source_dict(_get_source_row(source_id))


@router.post("/spaces/{space_id}/sources/upload")
async def upload_source(
    space_id: str, kind: Literal["pdf", "docx", "csv"] = Form(...), file: UploadFile = File(...)
) -> dict[str, object]:
    _space_row(space_id)
    source_id = new_id()
    dest_dir = Path(settings.upload_dir) / source_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (file.filename or "upload")
    dest.write_bytes(await file.read())

    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO sources (id, space_id, kind, name, uri, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, space_id, kind, file.filename or "upload", str(dest), now()),
        )
    threading.Thread(
        target=_run_ingest_background,
        args=(space_id, source_id, kind, file.filename or "upload", str(dest), None),
        daemon=True,
    ).start()
    return _source_dict(_get_source_row(source_id))


@router.delete("/sources/{source_id}")
def delete_source(source_id: str) -> dict[str, str]:
    row = _get_source_row(source_id)
    pipeline.delete_source(row["space_id"], source_id)
    with connect(settings.db_path) as conn:
        conn.execute("DELETE FROM sources WHERE id=?", (source_id,))
        conn.execute("DELETE FROM qa_cache WHERE space_id=?", (row["space_id"],))
    return {"status": "deleted"}


@router.get("/sources/{source_id}/trace")
def source_trace(source_id: str) -> dict[str, object]:
    row = _get_source_row(source_id)
    if not row["ingest_trace"]:
        raise HTTPException(status_code=404, detail="Not ingested yet")
    return json.loads(row["ingest_trace"])


@router.get("/sources/{source_id}/ingest/stream")
def source_ingest_stream(source_id: str) -> StreamingResponse:
    """Reconnect-safe by construction: it polls the sources row rather than owning the
    ingest work (a background thread does that), so a mid-ingest page refresh just
    re-attaches to whatever progress is already recorded."""

    def events() -> Iterator[str]:
        last_progress = None
        while True:
            with connect(settings.db_path) as conn:
                row = conn.execute(
                    "SELECT status, meta, ingest_trace, error FROM sources WHERE id=?", (source_id,)
                ).fetchone()
            if row is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'source not found'})}\n\n"
                return
            if row["status"] == "ready":
                yield f"data: {json.dumps({'type': 'complete', 'trace': json.loads(row['ingest_trace'])})}\n\n"
                return
            if row["status"] == "failed":
                yield f"data: {json.dumps({'type': 'error', 'message': row['error']})}\n\n"
                return
            progress = json.loads(row["meta"] or "{}").get("progress")
            if progress and progress != last_progress:
                last_progress = progress
                yield f"data: {json.dumps({'type': 'progress', **progress})}\n\n"
            time.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
