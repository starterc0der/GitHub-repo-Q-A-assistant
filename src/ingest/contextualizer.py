from __future__ import annotations

from dataclasses import replace

from src.index.schema import CodeChunk, FileSummary
from src.llm_client import LLMClient

MAX_CODE_CHARS_IN_PROMPT = 1500

SYSTEM_PROMPT = (
    "You write a one-sentence context header for a code chunk, to improve search "
    "retrieval. State the file's purpose and what this specific chunk does. "
    "Start with 'From {file}'. No preamble, one sentence."
)


class Contextualizer:
    """Prepends an LLM-written context header to each chunk before embedding.

    Improves retrieval on chunks whose code alone is ambiguous out of context
    (e.g. a short helper named `run`) — the header names the file and role.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def add_context_header(
        self, chunk: CodeChunk, file_summary: FileSummary, repo_summary: str
    ) -> CodeChunk:
        symbol_note = f" ({chunk.symbol_name})" if chunk.symbol_name else ""
        prompt = (
            f"Repo: {repo_summary}\n"
            f"File ({chunk.file_path}): {file_summary.summary}\n\n"
            f"Chunk{symbol_note}:\n```\n{chunk.code[:MAX_CODE_CHARS_IN_PROMPT]}\n```"
        )
        try:
            header = self.llm.complete(prompt, system=SYSTEM_PROMPT).strip()
        except RuntimeError:
            # A single flaky call must not abort ingestion of the whole repo.
            header = self._template_header(chunk, file_summary)
        return replace(chunk, context_header=header)

    def _template_header(self, chunk: CodeChunk, file_summary: FileSummary) -> str:
        symbol = f" {chunk.symbol_name}" if chunk.symbol_name else ""
        return f"From {chunk.file_path} ({file_summary.summary}). Defines{symbol}."
