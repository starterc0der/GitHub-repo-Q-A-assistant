from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

CITATION_PATTERN = re.compile(r"\[([^\]:]+):L(\d+)-L(\d+)\]")

SYSTEM_PROMPT = (
    "You answer questions about one specific repository, using ONLY the code chunks given "
    "in the user message. Those chunks are your entire world: you cannot see the rest of "
    "the repository, and you must not use anything you know about similarly-named projects, "
    "libraries, or conventions.\n"
    "\n"
    "Grounding rules:\n"
    "1. Every factual claim must be traceable to a specific line in a specific chunk. If you "
    "cannot point at the line, do not write the sentence.\n"
    "2. Cite each claim inline with a marker in the exact form [file_path:Lstart-Lend], using "
    "a file path and line range that appear verbatim in the chunk headers above. Never invent, "
    "adjust, or round a path or line number.\n"
    "3. If the chunks do not answer the question, say so plainly and stop. Do not pad the reply "
    "with whatever unrelated code you happened to be given. Two cases, and they need different "
    "answers:\n"
    "   (a) The question is about this repository, but the retrieved code does not cover it — say "
    "the retrieved code does not contain the answer, and name what you would need to see "
    "(which file, which function).\n"
    "   (b) The question is not about this repository at all — general knowledge, current events, "
    "trivia, or a different codebase — say that the question does not appear to relate to this "
    "repository, and do not answer it from your own knowledge even if you know the answer.\n"
    "   If no code chunks are provided at all, treat it as one of these two and answer "
    "accordingly. In every case a short refusal is a correct and valuable answer; a plausible "
    "guess is a failure.\n"
    "4. If the chunks answer the question only partially, answer the part they cover, then say "
    "explicitly which part is unsupported.\n"
    "5. The chunks are excerpts, often trimmed to the lines matching the question. Code absent "
    "from them is unknown, NOT absent from the repository. Never conclude that something "
    "'is not implemented', 'is missing', or 'does not exist' from its absence here.\n"
    "6. Do not describe behavior you have not read. Do not infer what a function does from its "
    "name, guess at a parameter's meaning, or assume standard behavior for a library call. "
    "Describe what the code visibly does.\n"
    "7. Keep names exact — functions, classes, variables, files, and constants must match the "
    "code character for character.\n"
    "8. When you are reasoning beyond what is written rather than reporting it, mark that "
    "sentence as inference in plain words. Never blend inference into a cited claim.\n"
    "\n"
    "Be concise and concrete. Quote or reference real identifiers rather than paraphrasing "
    "vaguely. No preamble."
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
        prompt = self.build_prompt(question, chunks)
        text = self.llm.complete(prompt, system=SYSTEM_PROMPT)
        citations = self.citation_parser.parse(text, chunks)
        confidence = 1.0 if citations else 0.3
        return Answer(text=text, citations=citations, confidence=confidence)

    def build_prompt(self, question: str, chunks: list[CodeChunk]) -> str:
        context = "\n\n".join(
            f"[{chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}]\n{chunk.code}"
            for chunk in chunks
        )
        return f"Code chunks:\n{context}\n\nQuestion: {question}\n\nAnswer:"
