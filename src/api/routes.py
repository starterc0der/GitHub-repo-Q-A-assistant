from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings
from src.pipeline import Pipeline

router = APIRouter()
pipeline = Pipeline(settings)


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
