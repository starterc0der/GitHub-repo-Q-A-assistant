from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LogicalFile:
    """One routable "file" for the summarize -> chunk -> embed pipeline.

    Repos produce many of these (one per source file, via the AST chunker); PDFs produce
    one per page; docx/pasted text produce exactly one covering the whole document. This
    is what lets the same per-file ingest loop in Pipeline serve all four source kinds.
    """

    name: str
    text: str


def load_pdf(path: Path, source_name: str) -> list[LogicalFile]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    files = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            files.append(LogicalFile(name=f"{source_name} · p.{i}", text=text))
    return files


def load_docx(path: Path, source_name: str) -> list[LogicalFile]:
    from docx import Document

    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [LogicalFile(name=source_name, text=text)] if text.strip() else []


def load_text(pasted: str, source_name: str) -> list[LogicalFile]:
    return [LogicalFile(name=source_name, text=pasted)] if pasted.strip() else []
