from __future__ import annotations

from src.generate.provenance import ClaimAttributor
from src.index.schema import CodeChunk


class FakeLLM:
    def __init__(self, response: str | None = None, fail: bool = False):
        self.response = response
        self.fail = fail
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("llm unreachable")
        return self.response


def _chunk(chunk_id: str, code: str, start_line: int = 1) -> CodeChunk:
    return CodeChunk(
        id=chunk_id, space_id="demo", source_id="src1", file_path="a.pdf",
        language="text", symbol_name=None, start_line=start_line,
        end_line=start_line + code.count("\n"), code=code,
    )


def test_attribute_maps_claims_to_chunk_ids_and_evidence() -> None:
    llm = FakeLLM(
        response='Draupadi\'s father is King Drupada. -> 1 | "Drupada is Draupadi\'s father."\n'
        'He ruled the kingdom of Panchala. -> 2 | Drupada ruled Panchala.'
    )
    attributor = ClaimAttributor(llm)
    chunks = [_chunk("c1", "Drupada is Draupadi's father."), _chunk("c2", "Drupada ruled Panchala.")]

    citations = attributor.attribute(
        "Draupadi's father is King Drupada. He ruled the kingdom of Panchala.", chunks
    )

    assert len(citations) == 2
    assert citations[0].claim == "Draupadi's father is King Drupada."
    assert citations[0].chunk_ids == ["c1"]
    assert citations[0].evidence == "Drupada is Draupadi's father."
    assert citations[0].evidence_chunk_id == "c1"
    assert citations[0].evidence_line == 1
    assert citations[1].chunk_ids == ["c2"]
    assert citations[1].evidence == "Drupada ruled Panchala."
    assert citations[1].evidence_chunk_id == "c2"
    assert citations[1].evidence_line == 1


def test_attribute_locates_evidence_line_across_wrapped_lines() -> None:
    """This corpus's prose chunks are hard-wrapped at a fixed width, so a quoted sentence
    routinely spans two physical lines — the locator must still find the line it STARTS
    on via word matching, not a failed substring search."""
    code = "Drupada, king of\nPanchala, was the father\nof Draupadi."
    llm = FakeLLM(response="Drupada was Draupadi's father. -> 1 | Panchala, was the father of Draupadi.")
    attributor = ClaimAttributor(llm)
    chunk = _chunk("c1", code, start_line=200)

    citations = attributor.attribute("Drupada was Draupadi's father.", [chunk])

    assert citations[0].evidence_chunk_id == "c1"
    assert citations[0].evidence_line == 201  # the wrap-line the quote actually starts on


def test_attribute_evidence_line_none_when_quote_is_paraphrased_not_verbatim() -> None:
    code = "Drupada, king of Panchala, was the father of Draupadi."
    llm = FakeLLM(response="Drupada was Draupadi's father. -> 1 | Draupadi's dad was King Drupada.")
    attributor = ClaimAttributor(llm)

    citations = attributor.attribute("Drupada was Draupadi's father.", [_chunk("c1", code)])

    assert citations[0].chunk_ids == ["c1"]  # attribution still holds
    assert citations[0].evidence_chunk_id == ""
    assert citations[0].evidence_line is None


def test_attribute_evidence_line_picks_the_chunk_that_actually_contains_it() -> None:
    """A claim cited against two chunks — the locator must find which ONE actually has
    the quote, not just default to the first."""
    llm = FakeLLM(response="A shared claim. -> 1, 2 | the real quote text")
    attributor = ClaimAttributor(llm)
    chunks = [_chunk("c1", "unrelated filler content", start_line=10), _chunk("c2", "the real quote text", start_line=50)]

    citations = attributor.attribute("A shared claim.", chunks)

    assert citations[0].evidence_chunk_id == "c2"
    assert citations[0].evidence_line == 50


def test_attribute_evidence_is_empty_when_llm_omits_it() -> None:
    """Old-style replies without the "| evidence" suffix still parse — chunk_ids/claim
    extraction doesn't depend on evidence being present."""
    llm = FakeLLM(response="Draupadi's father is King Drupada. -> 1")
    attributor = ClaimAttributor(llm)

    citations = attributor.attribute("Draupadi's father is King Drupada.", [_chunk("c1", "x")])

    assert citations[0].chunk_ids == ["c1"]
    assert citations[0].evidence == ""


def test_attribute_unsupported_claim_has_no_evidence() -> None:
    llm = FakeLLM(response="An unsupported claim. -> none")
    attributor = ClaimAttributor(llm)

    citations = attributor.attribute("An unsupported claim.", [_chunk("c1", "x")])

    assert citations[0].chunk_ids == []
    assert citations[0].evidence == ""


def test_attribute_handles_multiple_chunk_ids_and_none() -> None:
    llm = FakeLLM(response="A shared claim. -> 1, 2\nAn unsupported claim. -> none")
    attributor = ClaimAttributor(llm)
    chunks = [_chunk("c1", "x"), _chunk("c2", "y")]

    citations = attributor.attribute("A shared claim. An unsupported claim.", chunks)

    assert citations[0].chunk_ids == ["c1", "c2"]
    assert citations[1].chunk_ids == []


def test_attribute_skips_llm_call_with_no_chunks_or_empty_answer() -> None:
    llm = FakeLLM(response="unused")
    attributor = ClaimAttributor(llm)

    assert attributor.attribute("an answer", []) == []
    assert attributor.attribute("", [_chunk("c1", "x")]) == []
    assert llm.calls == 0


def test_attribute_fails_open_to_empty_list_on_llm_failure() -> None:
    llm = FakeLLM(fail=True)
    attributor = ClaimAttributor(llm)

    assert attributor.attribute("an answer", [_chunk("c1", "x")]) == []


def test_attribute_ignores_out_of_range_chunk_numbers() -> None:
    llm = FakeLLM(response="A claim. -> 99")
    attributor = ClaimAttributor(llm)

    citations = attributor.attribute("A claim.", [_chunk("c1", "x")])

    assert citations[0].chunk_ids == []
