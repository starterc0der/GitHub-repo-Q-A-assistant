from __future__ import annotations

from dataclasses import dataclass, field

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

SUPPORTED_SENTINEL = "SUPPORTED"

SYSTEM_PROMPT = (
    "You check whether an AI-generated answer is fully grounded in the given source "
    "chunks — nothing else. Do not judge whether the answer is well-written or complete, "
    "only whether every factual claim it makes is actually stated or directly implied by "
    "the chunks. If the answer explicitly says information is missing (e.g. \"the chunks "
    "don't mention this\"), that is not an unsupported claim.\n"
    f"If every claim is grounded, reply with exactly: {SUPPORTED_SENTINEL}\n"
    "Otherwise, reply with one unsupported claim per line, quoting the claim from the "
    "answer verbatim. No other text, no explanation."
)


@dataclass
class FaithfulnessResult:
    supported: bool
    unsupported_claims: list[str] = field(default_factory=list)
    # True when the LLM call itself failed — supported=True in that case (fail open,
    # same as every other LLM-backed check in this codebase), but callers that want to
    # distinguish "checked and clean" from "couldn't check" can look here.
    checked: bool = True


class FaithfulnessChecker:
    """Catches claims a generated answer makes that the retrieved chunks don't actually
    support — the thing citation parsing used to guard against before it was removed.
    Answers are free-form prose now, so this checks the whole answer against the whole
    context rather than validating per-claim citation markers."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def check(self, question: str, answer_text: str, chunks: list[CodeChunk]) -> FaithfulnessResult:
        if not answer_text.strip() or not chunks:
            return FaithfulnessResult(supported=True)

        context = "\n\n".join(
            f"[{chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}]\n{chunk.code}"
            for chunk in chunks
        )
        prompt = f"Source chunks:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer_text}"
        try:
            reply = self.llm.complete(prompt, system=SYSTEM_PROMPT).strip()
        except RuntimeError:
            return FaithfulnessResult(supported=True, checked=False)

        if not reply or reply == SUPPORTED_SENTINEL:
            return FaithfulnessResult(supported=True)
        claims = [line.strip("-•* \t") for line in reply.splitlines() if line.strip()]
        return FaithfulnessResult(supported=False, unsupported_claims=claims)
