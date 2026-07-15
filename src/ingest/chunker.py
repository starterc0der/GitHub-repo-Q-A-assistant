from __future__ import annotations

from pathlib import Path

from src.index.schema import CodeChunk

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}


class Tokenizer:
    """Approximate token counting and size-bounded text splitting.

    No model-specific tokenizer dependency — a char/4 heuristic is close
    enough for chunk-sizing decisions and works for any local LLM.
    """

    CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN if text else 0

    def split(self, text: str, max_chars: int, overlap_chars: int) -> list[str]:
        return [block for block, _, _ in self._split_lines(text, max_chars, overlap_chars)]

    def split_with_lines(
        self, text: str, max_chars: int, overlap_chars: int
    ) -> list[tuple[str, int, int]]:
        """Like split(), but also returns each block's 1-indexed (start_line, end_line).

        Blocks are assembled from whole lines only, so boundaries always land on
        line breaks — this is what lets citations stay byte-for-byte accurate.
        """
        return self._split_lines(text, max_chars, overlap_chars)

    def _split_lines(
        self, text: str, max_chars: int, overlap_chars: int
    ) -> list[tuple[str, int, int]]:
        lines = text.splitlines(keepends=True)
        n = len(lines)
        blocks: list[tuple[str, int, int]] = []
        i = 0
        while i < n:
            size = 0
            j = i
            while j < n and (j == i or size + len(lines[j]) <= max_chars):
                size += len(lines[j])
                j += 1
            blocks.append(("".join(lines[i:j]), i + 1, j))
            if j >= n:
                break
            overlap_size = 0
            k = j
            while k > i + 1 and overlap_size < overlap_chars:
                k -= 1
                overlap_size += len(lines[k])
            i = k
        return blocks


class RecursiveChunker:
    """Naive line-boundary chunker — the M1 baseline, kept for eval comparison."""

    def __init__(self, tokenizer: Tokenizer, max_chars: int = 2000, overlap_chars: int = 200):
        self.tokenizer = tokenizer
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_file(self, path: Path, root: Path, repo: str) -> list[CodeChunk]:
        text = path.read_text(encoding="utf-8", errors="replace")
        language = LANGUAGE_BY_EXT.get(path.suffix, "text")
        file_path = str(path.relative_to(root))
        chunks = []
        for block, start_line, end_line in self.tokenizer.split_with_lines(
            text, self.max_chars, self.overlap_chars
        ):
            chunks.append(
                CodeChunk(
                    id=f"{file_path}::{start_line}-{end_line}",
                    repo=repo,
                    file_path=file_path,
                    language=language,
                    symbol_name=None,
                    start_line=start_line,
                    end_line=end_line,
                    code=block,
                )
            )
        return chunks
