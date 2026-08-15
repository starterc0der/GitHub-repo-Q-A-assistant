from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _locate_line(evidence: str, chunk: CodeChunk) -> int | None:
    """Finds the source line where `evidence` actually appears in `chunk.code`, by
    matching its word sequence against a (word -> line index) token stream built from the
    chunk — not a plain substring search, since this corpus's prose chunks are hard-wrapped
    at a fixed width, so a quoted sentence routinely spans multiple physical lines and
    would never match as one contiguous substring. Punctuation-insensitive, case-insensitive.
    Returns None (not a guess) when the quote can't be matched exactly — e.g. the model
    paraphrased instead of quoting verbatim."""
    needle = _words(evidence)
    if not needle:
        return None
    lines = chunk.code.split("\n")
    tokens: list[tuple[str, int]] = [
        (word, i) for i, line in enumerate(lines) for word in _words(line)
    ]
    n = len(needle)
    for start in range(len(tokens) - n + 1):
        if all(tokens[start + k][0] == needle[k] for k in range(n)):
            return chunk.start_line + tokens[start][1]
    return None

CLAIM_ATTRIBUTION_SYSTEM_PROMPT = (
    "You verify each factual claim in an AI-generated answer against numbered source "
    "chunks, and attribute it to the chunk(s) that actually support it. This is a real "
    "check, not a loose topic match: a chunk only counts as support if it actually states "
    "or directly implies the claim — a chunk merely mentioning the same person/topic "
    "without confirming the specific claim does NOT count.\n"
    "Break the answer into its individual factual claims — one per distinct fact, NOT one "
    "per sentence: a single sentence stating two facts via a comma or relative clause (e.g. "
    "\"X is Y, who did Z\") is TWO separate claims, \"X is Y\" and \"who did Z\", each "
    "verified and cited on its own — do not merge them just because they share a sentence. "
    "Skip meta-sentences that state a gap or limitation rather than a fact (e.g. \"the "
    "source material does not say X\").\n"
    "For each claim, reply with one line in this exact format: the claim quoted verbatim "
    "from the answer, then \" -> \", then the verified chunk number(s) comma-separated, "
    "then \" | \", then a short verbatim quote (under 25 words, copied exactly from that "
    "chunk) that is the actual evidence proving THIS claim specifically — not a paraphrase, "
    "and not the evidence for a different claim in the answer.\n"
    "Example, for the answer \"Draupadi's father is King Drupada, who ruled the kingdom of "
    "Panchala.\":\n"
    "Draupadi's father is King Drupada. -> 2 | Drupada was Draupadi's father.\n"
    "who ruled the kingdom of Panchala. -> 2 | Drupada, king of Panchala.\n"
    "If a claim is not actually verified by any chunk — unsupported, or only loosely "
    "related — use \" -> none\" with nothing after it.\n"
    "No other text, no preamble, no numbering of your own beyond what's shown."
)


@dataclass
class ClaimCitation:
    claim: str
    chunk_ids: list[str] = field(default_factory=list)
    # Short verbatim quote from the cited chunk(s) — the actual sentence/line that proves
    # the claim, not just "which chunk". Empty when unverified (chunk_ids == []).
    evidence: str = ""
    # Which cited chunk the evidence quote was actually located in, and the exact source
    # line it starts on — computed locally via _locate_line, not reported by the LLM
    # (which is unreliable at line-number arithmetic). Empty/None when the quote couldn't
    # be matched verbatim against any cited chunk.
    evidence_chunk_id: str = ""
    evidence_line: int | None = None


class ClaimAttributor:
    """Does two things in one pass, computed after the fact rather than self-reported by
    the answer model (unreliable, and the whole reason the old inline-citation approach
    was dropped from the answer prompt — see src/generate/answer.py's SYSTEM_PROMPT):

    - Provenance: maps each claim back to the chunk(s) that actually support it.
    - Verification: a claim with no supporting chunk (chunk_ids == []) IS an unverified /
      possibly-hallucinated claim — the same "-> none" signal serves both jobs, so
      verification costs nothing beyond the citation call already being made.

    Chat stays plain prose; this is surfaced only in the pipeline trace view."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def attribute(self, answer_text: str, chunks: list[CodeChunk]) -> list[ClaimCitation]:
        if not answer_text.strip() or not chunks:
            return []
        numbered = "\n\n".join(
            f"[{i}] {chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}\n{chunk.code}"
            for i, chunk in enumerate(chunks, start=1)
        )
        prompt = f"Source chunks:\n{numbered}\n\nAnswer:\n{answer_text}"
        try:
            reply = self.llm.complete(prompt, system=CLAIM_ATTRIBUTION_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return []

        by_id = {chunk.id: chunk for chunk in chunks}
        citations = []
        for line in reply.splitlines():
            if "->" not in line:
                continue
            claim, rest = line.rsplit("->", 1)
            claim = claim.strip()
            if not claim:
                continue
            ids_part, _, evidence_part = rest.partition("|")
            chunk_ids = []
            for token in ids_part.split(","):
                token = token.strip()
                if token.isdigit() and 1 <= int(token) <= len(chunks):
                    chunk_ids.append(chunks[int(token) - 1].id)
            evidence = evidence_part.strip().strip('"“”') if chunk_ids else ""

            evidence_chunk_id, evidence_line = "", None
            for cid in chunk_ids:
                located = _locate_line(evidence, by_id[cid])
                if located is not None:
                    evidence_chunk_id, evidence_line = cid, located
                    break

            citations.append(
                ClaimCitation(claim, chunk_ids, evidence, evidence_chunk_id, evidence_line)
            )
        return citations
