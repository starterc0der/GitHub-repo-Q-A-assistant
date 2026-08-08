from __future__ import annotations

from pathlib import Path

from src.ingest.ast_chunker import ASTChunker
from src.ingest.chunker import RecursiveChunker, Tokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "sample.py"


def _chunker() -> ASTChunker:
    tokenizer = Tokenizer()
    return ASTChunker(tokenizer, RecursiveChunker(tokenizer, max_chars=2000, overlap_chars=200))


def test_chunk_file_splits_on_definitions_with_exact_line_ranges() -> None:
    chunker = _chunker()
    original_lines = FIXTURE.read_text().splitlines()

    chunks = chunker.chunk_file(FIXTURE, FIXTURE.parent, space_id="s", source_id="fixtures", language="python")

    assert [c.symbol_name for c in chunks] == [None, "add", "subtract", "Calculator"]
    for chunk in chunks:
        expected = "\n".join(original_lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.code == expected


def test_decorated_function_unwraps_to_inner_definition_name(tmp_path: Path) -> None:
    path = tmp_path / "deco.py"
    path.write_text('@staticmethod\ndef greet():\n    return "hi"\n')
    chunker = _chunker()

    chunks = chunker.chunk_file(path, tmp_path, space_id="s", source_id="demo", language="python")

    assert len(chunks) == 1
    assert chunks[0].symbol_name == "greet"
    assert chunks[0].code.startswith("@staticmethod")


def test_unsupported_language_falls_back_to_recursive_chunker() -> None:
    chunker = _chunker()

    chunks = chunker.chunk_file(FIXTURE, FIXTURE.parent, space_id="s", source_id="fixtures", language="text")

    assert chunks
    assert all(c.symbol_name is None for c in chunks)


def test_empty_file_returns_no_chunks(tmp_path: Path) -> None:
    path = tmp_path / "empty.py"
    path.write_text("")
    chunker = _chunker()

    assert chunker.chunk_file(path, tmp_path, space_id="s", source_id="demo", language="python") == []
