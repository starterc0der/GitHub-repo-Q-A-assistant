from __future__ import annotations

from pathlib import Path

from src.ingest.chunker import RecursiveChunker, Tokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "sample.py"


def test_tokenizer_split_respects_max_chars_with_overlap() -> None:
    text = "".join(f"line {i}\n" for i in range(20))
    tokenizer = Tokenizer()

    blocks = tokenizer.split(text, max_chars=30, overlap_chars=10)

    assert len(blocks) > 1
    assert all(len(block) <= 40 for block in blocks)
    assert "".join(blocks).count("line 0\n") >= 1


def test_tokenizer_count_is_roughly_chars_over_four() -> None:
    tokenizer = Tokenizer()

    assert tokenizer.count("") == 0
    assert tokenizer.count("abcd") == 1
    assert tokenizer.count("abcdefgh") == 2


def test_chunk_file_line_ranges_are_exact() -> None:
    tokenizer = Tokenizer()
    chunker = RecursiveChunker(tokenizer, max_chars=100, overlap_chars=20)
    original_lines = FIXTURE.read_text().splitlines(keepends=True)

    chunks = chunker.chunk_file(FIXTURE, FIXTURE.parent, space_id="s", source_id="fixtures")

    assert len(chunks) > 1
    for chunk in chunks:
        expected = "".join(original_lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.code == expected
        assert chunk.file_path == "sample.py"
        assert chunk.space_id == "s"
        assert chunk.source_id == "fixtures"


def test_chunk_text_tags_chunks_with_space_and_source() -> None:
    tokenizer = Tokenizer()
    chunker = RecursiveChunker(tokenizer, max_chars=1000, overlap_chars=0)

    chunks = chunker.chunk_text("hello\nworld\n", "Handbook.pdf · p.1", "space1", "source1")

    assert len(chunks) == 1
    assert chunks[0].space_id == "space1"
    assert chunks[0].source_id == "source1"
    assert chunks[0].file_path == "Handbook.pdf · p.1"
    assert chunks[0].language == "text"
