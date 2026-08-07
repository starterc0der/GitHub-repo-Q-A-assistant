from __future__ import annotations

import re
from dataclasses import replace

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You identify which lines of each code chunk are relevant to a question. Chunks are "
    "numbered, and lines within each chunk are numbered starting at 1.\n"
    "Reply with exactly one line per chunk, in this form:\n"
    "  <chunk number>: <comma-separated relevant line numbers>\n"
    "or, if nothing in that chunk is relevant:\n"
    "  <chunk number>: NONE\n"
    "Cover every chunk you were given, in order. No other text, no explanation."
)

LINE_NUMBER_PATTERN = re.compile(r"\d+")
# Models drift on the separator ("3: 12" / "3 - NONE" / "3) 12"), so accept any of them.
VERDICT_PATTERN = re.compile(r"^\s*(\d+)\s*[:.\-)]\s*(.+?)\s*$")


class Compressor:
    """Trims each chunk to only its question-relevant lines before generation.

    The LLM returns line *numbers*, not reformatted text, so exact start_line/
    end_line offsets (needed for citations) survive the transform. Compression
    narrows to the contiguous min..max window rather than splicing non-adjacent
    lines — a deliberate simplification, see PROJECT_SPEC.md §5.

    All chunks go in one request: per-chunk calls made compression the biggest consumer
    of both latency and API quota in a query.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compress(self, question: str, chunk: CodeChunk) -> CodeChunk | None:
        """Single-chunk wrapper, so both paths share one implementation."""
        return self.compress_batch(question, [chunk])[0]

    def compress_batch(self, question: str, chunks: list[CodeChunk]) -> list[CodeChunk | None]:
        """Compress every chunk in one call. Returns one entry per input chunk, in order:
        a trimmed chunk, or None when the model judged it irrelevant."""
        if not chunks:
            return []

        blocks = []
        for i, chunk in enumerate(chunks, 1):
            numbered = "\n".join(
                f"{n + 1}: {line}" for n, line in enumerate(chunk.code.splitlines())
            )
            blocks.append(f"Chunk {i} ({chunk.file_path}):\n{numbered}")
        prompt = f"Question: {question}\n\n" + "\n\n".join(blocks)

        try:
            response = self.llm.complete(prompt, system=SYSTEM_PROMPT)
        except RuntimeError:
            # Can't compress right now — keep every chunk whole rather than drop context.
            return list(chunks)

        verdicts = self._parse(response, len(chunks))
        return [
            # Unjudged is not irrelevant: a truncated reply must not drop the tail.
            chunk if i not in verdicts else self._apply(chunk, verdicts[i])
            for i, chunk in enumerate(chunks, 1)
        ]

    def _parse(self, response: str, count: int) -> dict[int, list[int] | None]:
        """Map chunk number -> relevant line numbers, or None for an explicit NONE."""
        verdicts: dict[int, list[int] | None] = {}
        for line in response.splitlines():
            match = VERDICT_PATTERN.match(line)
            if not match:
                continue
            index = int(match.group(1))
            if not 1 <= index <= count:
                continue
            body = match.group(2)
            verdicts[index] = None if body.upper().startswith("NONE") else [
                int(n) for n in LINE_NUMBER_PATTERN.findall(body)
            ]
        return verdicts

    def _apply(self, chunk: CodeChunk, relevant: list[int] | None) -> CodeChunk | None:
        if relevant is None:
            return None
        lines = chunk.code.splitlines()
        kept = sorted({n for n in relevant if 1 <= n <= len(lines)})
        if not kept:
            return None
        lo, hi = kept[0], kept[-1]
        return replace(
            chunk,
            code="\n".join(lines[lo - 1 : hi]),
            start_line=chunk.start_line + lo - 1,
            end_line=chunk.start_line + hi - 1,
        )
