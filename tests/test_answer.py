from __future__ import annotations

from src.generate.answer import CitationParser
from src.index.schema import CodeChunk


def _chunk(file_path: str, start_line: int, end_line: int, code: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::{start_line}-{end_line}",
        repo="demo",
        file_path=file_path,
        language="python",
        symbol_name=None,
        start_line=start_line,
        end_line=end_line,
        code=code,
    )


def test_parse_recovers_file_range_and_snippet() -> None:
    chunk = _chunk("src/util.py", 1, 5, "def a():\n    pass\n\ndef b():\n    pass\n")
    text = "The function is defined here [src/util.py:L4-L5]."

    citations = CitationParser().parse(text, [chunk])

    assert len(citations) == 1
    citation = citations[0]
    assert citation.file_path == "src/util.py"
    assert citation.start_line == 4
    assert citation.end_line == 5
    assert citation.snippet == "def b():\n    pass"


def test_parse_returns_empty_list_when_no_markers_present() -> None:
    citations = CitationParser().parse("I don't know based on the given code.", [])

    assert citations == []
