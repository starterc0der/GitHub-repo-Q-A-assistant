from __future__ import annotations

import re
from dataclasses import replace

from src.index.schema import CodeChunk, FileSummary
from src.llm_client import LLMClient

MAX_CODE_CHARS_IN_PROMPT = 1500

SYSTEM_PROMPT = (
    "You write a one-sentence context header for each numbered code chunk below, to "
    "improve search retrieval. State the file's purpose and what that specific chunk "
    "does. Reply with exactly one line per chunk, in the form:\n"
    "  <chunk number>: <header starting with 'From {file}'>\n"
    "Cover every chunk you were given, in order. No other text, no explanation."
)

VERDICT_PATTERN = re.compile(r"^\s*(\d+)\s*[:.\-)]\s*(.+?)\s*$")


class Contextualizer:
    """Prepends a context header to each chunk before embedding.

    Improves retrieval on chunks whose code alone is ambiguous out of context
    (e.g. a short helper named `run`) — the header names the file and role.

    Prose chunks (PDFs, docs, pasted text) skip the LLM: a paragraph is already
    self-describing, so a templated "From {name}, page N" header is free and
    just as useful as a restatement.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def add_context_header(
        self, chunk: CodeChunk, file_summary: FileSummary, repo_summary: str
    ) -> CodeChunk:
        return self.add_context_headers_batch([chunk], file_summary, repo_summary)[0]

    def add_context_headers_batch(
        self, chunks: list[CodeChunk], file_summary: FileSummary, repo_summary: str
    ) -> list[CodeChunk]:
        """One LLM call per file for every chunk in it, instead of one call per chunk."""
        if not chunks:
            return []
        if chunks[0].language == "text":
            return [self._template(chunk, file_summary) for chunk in chunks]

        blocks = [
            f"Chunk {i}" + (f" ({chunk.symbol_name})" if chunk.symbol_name else "") +
            f":\n```\n{chunk.code[:MAX_CODE_CHARS_IN_PROMPT]}\n```"
            for i, chunk in enumerate(chunks, 1)
        ]
        prompt = (
            f"Repo: {repo_summary}\n"
            f"File ({file_summary.file_path}): {file_summary.summary}\n\n" + "\n\n".join(blocks)
        )
        try:
            response = self.llm.complete(prompt, system=SYSTEM_PROMPT)
        except RuntimeError:
            # A flaky call must not abort ingestion of the whole repo.
            return [self._template(chunk, file_summary) for chunk in chunks]

        headers = self._parse(response, len(chunks))
        return [
            replace(chunk, context_header=headers.get(i) or self._template_text(chunk, file_summary))
            for i, chunk in enumerate(chunks, 1)
        ]

    def _parse(self, response: str, count: int) -> dict[int, str]:
        headers: dict[int, str] = {}
        for line in response.splitlines():
            match = VERDICT_PATTERN.match(line)
            if not match:
                continue
            index = int(match.group(1))
            if 1 <= index <= count:
                headers[index] = match.group(2)
        return headers

    def _template(self, chunk: CodeChunk, file_summary: FileSummary) -> CodeChunk:
        return replace(chunk, context_header=self._template_text(chunk, file_summary))

    def _template_text(self, chunk: CodeChunk, file_summary: FileSummary) -> str:
        symbol = f" {chunk.symbol_name}" if chunk.symbol_name else ""
        return f"From {chunk.file_path} ({file_summary.summary}). Defines{symbol}."
