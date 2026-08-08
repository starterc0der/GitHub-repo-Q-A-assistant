from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeChunk:
    """A retrievable slice of a source (a repo file, a PDF page, a docx, pasted text).

    file_path doubles as the citation label: a real relative path for code, or a
    human-readable location like "Handbook.pdf · p.14" for prose sources — both render
    identically as "{file_path}:L{start}-L{end}", so citation parsing needs no special case.
    """

    id: str
    space_id: str
    source_id: str
    file_path: str
    language: str
    symbol_name: str | None
    start_line: int
    end_line: int
    code: str
    context_header: str = ""

    @property
    def embeddable_text(self) -> str:
        if not self.context_header:
            return self.code
        return f"{self.context_header}\n{self.code}"


@dataclass
class FileSummary:
    """A one-per-logical-file summary used by the routing layer. One "file" is one repo
    file, one PDF page, or the whole document for docx/pasted text."""

    space_id: str
    source_id: str
    file_path: str
    language: str
    summary: str
    symbols: list[str] = field(default_factory=list)
