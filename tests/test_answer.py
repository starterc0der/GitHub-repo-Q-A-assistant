from __future__ import annotations

from src.generate.answer import CitationParser
from src.index.schema import CodeChunk


def _chunk(file_path: str, start_line: int, end_line: int, code: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::{start_line}-{end_line}",
        space_id="demo",
        source_id="src1",
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


def test_parse_splits_a_grouped_bracket_into_separate_citations() -> None:
    """Models sometimes combine ranges into one bracket despite the prompt asking for
    one marker per claim — this must still resolve rather than silently matching nothing."""
    chunk = _chunk("src/pipeline.py", 60, 90, "\n".join(f"line{i}" for i in range(60, 91)))
    text = "Two things happen here [src/pipeline.py:L62-L62, L87-L87]."

    citations = CitationParser().parse(text, [chunk])

    assert len(citations) == 2
    assert [(c.start_line, c.end_line) for c in citations] == [(62, 62), (87, 87)]
    assert all(c.file_path == "src/pipeline.py" for c in citations)


def test_parse_accepts_a_bare_line_number_with_no_dash() -> None:
    """Models sometimes cite a single row as [file:L5] instead of [file:L5-L5] — common
    for one-row citations like a CSV row. Must still resolve, not silently match nothing."""
    chunk = _chunk("cars.csv", 472, 472, "Nissan Ariya,electric,,,")
    text = "The Nissan Ariya is electric [cars.csv:L472]."

    citations = CitationParser().parse(text, [chunk])

    assert len(citations) == 1
    assert (citations[0].start_line, citations[0].end_line) == (472, 472)


def test_parse_accepts_a_mix_of_bare_and_ranged_lines_in_one_bracket() -> None:
    chunk = _chunk("cars.csv", 100, 200, "\n".join(f"row{i}" for i in range(100, 201)))
    text = "See these rows [cars.csv:L105, L150-L152]."

    citations = CitationParser().parse(text, [chunk])

    assert [(c.start_line, c.end_line) for c in citations] == [(105, 105), (150, 152)]
