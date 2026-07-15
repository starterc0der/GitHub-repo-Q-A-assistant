from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeChunk:
    """A retrievable slice of a source file."""

    id: str
    repo: str
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
    """A one-per-file summary used by the routing layer."""

    repo: str
    file_path: str
    language: str
    summary: str
    symbols: list[str] = field(default_factory=list)
