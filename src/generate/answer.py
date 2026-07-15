from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

CITATION_PATTERN = re.compile(r"\[([^\]:]+):L(\d+)-L(\d+)\]")

SYSTEM_PROMPT = (
    "You are a code assistant answering questions about a specific repository. "
    "Answer only using the provided code chunks — if they don't contain the answer, say so. "
    "Cite every claim with a marker in the exact form [file_path:Lstart-Lend] referencing the "
    "chunk it came from."
)


@dataclass
class Citation:
    file_path: str
    start_line: int
    end_line: int
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 1.0


class CitationParser:
    """Extracts [file:Lstart-Lend] markers from LLM output and matches them to chunks."""

    def parse(self, text: str, chunks: list[CodeChunk]) -> list[Citation]:
        citations = []
        for file_path, start_str, end_str in CITATION_PATTERN.findall(text):
            start_line, end_line = int(start_str), int(end_str)
            chunk = self._find_chunk(file_path, start_line, chunks)
            if chunk is None:
                continue
            citations.append(
                Citation(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    snippet=self._extract_snippet(chunk, start_line, end_line),
                )
            )
        return citations

    def _find_chunk(
        self, file_path: str, start_line: int, chunks: list[CodeChunk]
    ) -> CodeChunk | None:
        for chunk in chunks:
            if chunk.file_path == file_path and chunk.start_line <= start_line <= chunk.end_line:
                return chunk
        return None

    def _extract_snippet(self, chunk: CodeChunk, start_line: int, end_line: int) -> str:
        lines = chunk.code.splitlines()
        offset = max(0, start_line - chunk.start_line)
        count = max(1, end_line - start_line + 1)
        return "\n".join(lines[offset : offset + count])


class AnswerGenerator:
    """Turns retrieved chunks + a question into a cited, grounded answer."""

    def __init__(self, llm: LLMClient, citation_parser: CitationParser):
        self.llm = llm
        self.citation_parser = citation_parser

    def answer(self, question: str, chunks: list[CodeChunk]) -> Answer:
        prompt = self._build_prompt(question, chunks)
        text = self.llm.complete(prompt, system=SYSTEM_PROMPT)
        citations = self.citation_parser.parse(text, chunks)
        confidence = 1.0 if citations else 0.3
        return Answer(text=text, citations=citations, confidence=confidence)

    def _build_prompt(self, question: str, chunks: list[CodeChunk]) -> str:
        context = "\n\n".join(
            f"[{chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}]\n{chunk.code}"
            for chunk in chunks
        )
        return f"Code chunks:\n{context}\n\nQuestion: {question}\n\nAnswer:"
