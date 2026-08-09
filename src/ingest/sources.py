from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.index.schema import CodeChunk


@dataclass
class LogicalFile:
    """One routable "file" for the summarize -> chunk -> embed pipeline.

    Repos produce many of these (one per source file, via the AST chunker); PDFs produce
    one per page; docx/pasted text/csv produce exactly one covering the whole document.
    This is what lets the same per-file ingest loop in Pipeline serve every source kind.
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


def load_csv(path: Path, source_name: str) -> list[LogicalFile]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [LogicalFile(name=source_name, text=text)] if text.strip() else []


def chunk_csv(
    text: str, file_path: str, space_id: str, source_id: str, rows_per_chunk: int = 50
) -> list[CodeChunk]:
    """Batches real rows (never splits one) and stamps the header into context_header, not
    code, so citations still point at real row numbers. Line-based, not a real CSV parser —
    a quoted newline inside a field would split wrong; fine for typical simple exports."""
    lines = text.splitlines()
    if not lines:
        return []
    header, data_lines = lines[0], lines[1:]
    chunks = []
    for start in range(0, len(data_lines), rows_per_chunk):
        batch = data_lines[start : start + rows_per_chunk]
        if not any(line.strip() for line in batch):
            continue
        start_line = start + 2  # +1 for the header row, +1 for 1-indexing
        end_line = start_line + len(batch) - 1
        chunks.append(
            CodeChunk(
                id=f"{file_path}::{start_line}-{end_line}",
                space_id=space_id,
                source_id=source_id,
                file_path=file_path,
                language="csv",
                symbol_name=None,
                start_line=start_line,
                end_line=end_line,
                code="\n".join(batch),
                context_header=header,
            )
        )
    return chunks
