from __future__ import annotations

import re
from dataclasses import replace

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You identify which lines of a code chunk are relevant to a question. "
    "Lines are numbered starting at 1. Reply with ONLY a comma-separated list of "
    "relevant line numbers (e.g. '3,4,5,9'), or 'NONE' if nothing is relevant. "
    "No other text."
)

LINE_NUMBER_PATTERN = re.compile(r"\d+")


class Compressor:
    """Trims each chunk to only its question-relevant lines before generation.

    The LLM returns line *numbers*, not reformatted text, so exact start_line/
    end_line offsets (needed for citations) survive the transform. Compression
    narrows to the contiguous min..max window rather than splicing non-adjacent
    lines — a deliberate simplification, see PROJECT_SPEC.md §5.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compress(self, question: str, chunk: CodeChunk) -> CodeChunk | None:
        lines = chunk.code.splitlines()
        numbered = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
        prompt = f"Question: {question}\n\nCode:\n{numbered}"
        try:
            response = self.llm.complete(prompt, system=SYSTEM_PROMPT).strip()
        except RuntimeError:
            return chunk  # can't compress right now, keep the whole chunk rather than drop it

        if response.upper().startswith("NONE"):
            return None

        relevant = sorted(
            {int(n) for n in LINE_NUMBER_PATTERN.findall(response) if 1 <= int(n) <= len(lines)}
        )
        if not relevant:
            return None

        lo, hi = min(relevant), max(relevant)
        return replace(
            chunk,
            code="\n".join(lines[lo - 1 : hi]),
            start_line=chunk.start_line + lo - 1,
            end_line=chunk.start_line + hi - 1,
        )
