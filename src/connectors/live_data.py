from __future__ import annotations

import json
import re
from dataclasses import dataclass

import redis

from src.config import settings
from src.crypto import decrypt
from src.db import connect

# ULB name (as it appears in the place doc / place.ulb column) -> the Redis key prefix
# used for that ULB's devices. One entry per city as it comes online.
ULB_REDIS_PREFIX = {
    "cuttack": "ctc",
}

# The model sometimes writes ```json out of habit despite rule 1 below spelling out
# ```table — accept either fence label rather than silently degrading to a raw JSON
# blob leaking into the visible answer text.
TABLE_PATTERN = re.compile(r"```(?:table|json)\s*\n(.*?)\n```", re.DOTALL)

# Handed to the generation LLM alongside the already-extracted readings (see
# build_live_data_context) — each sub-place's pressure/flow/totalizer is already
# correctly matched to it by construction (one Redis key = one sub-place's own data), so
# this only needs to cover phrasing/semantics the LLM can't get from the numbers alone.
LIVE_DATA_SYSTEM_PROMPT = (
    "You are answering from live sensor readings for one or more sub-places at one or "
    "more places, already extracted and labeled for you (location, pressure, flow, "
    "totalizer per sub-place, plus level and chlorine where they apply — see rules 8 "
    "and 9).\n"
    "\n"
    "A question naming a place or a device id — including 'show me data for X', 'data "
    "for device X', 'status of X' — is asking for these LIVE READINGS (pressure, flow, "
    "totalizer), never for the device's catalog/mapping details (its pid, its parent "
    "place, its sub-place list). Only describe mapping details if the question "
    "explicitly asks what place a device belongs to, or similar — not as a substitute "
    "for the readings.\n"
    "\n"
    "Rules:\n"
    '1. Default to a table: start your answer with a fenced ```table block (before any '
    'prose) of this exact JSON shape: {"columns": [...], "rows": [[...], ...]}, one row '
    "per sub-place, values as plain numbers or short strings, same column order in "
    "every row. Only skip the table if the user's question explicitly asks for a list, "
    "a single number, or some other specific format instead.\n"
    "2. Never call it a 'point' — it's a sub-place. Label each row's location as "
    "'<location> (<inlet or outlet>)', e.g. 'ESR (inlet)', 'Deer_Park (outlet)' — never "
    "'inlet at ESR' or similar.\n"
    "3. Do not include the device id or pid as a column, and do not mention either in "
    "prose, unless the user's question explicitly asks for the device id or pid.\n"
    "4. Do not explain what pressure/flow/totalizer mean, and do not say whether a "
    "totalizer represents water loaded or consumed — the user wants the data, not an "
    "explanation of it. A totalizer resets to 0 at midnight, so its current value "
    "already is today's figure; never call it a lifetime total and never compute a "
    "delta.\n"
    "5. Never state or imply a unit (no 'bar', 'psi', 'LPS', 'm³', etc.) — units are not "
    "confirmed. Report the bare number only.\n"
    "6. If a sub-place's reading is marked unavailable, say so plainly (as "
    "'unavailable' in the table, or in prose) — never invent or estimate a value.\n"
    "7. If the question names a specific place or sub-place, answer only for that one "
    "unless asked for more.\n"
    "8. 'level' appears only on the inlet's reading, never on an outlet's — it's how "
    "full the inlet's storage/tank is, not a pressure or flow value. Include it as its "
    "own column/value only for the inlet row; do not put it on outlet rows.\n"
    "9. 'Chlorine (overall for this place...)' is a single place-wide reading, not "
    "per-sub-place — it appears once per place because every outlet reports the same "
    "value. Report it once (e.g. one extra table row or one sentence), never repeated "
    "as a column on every outlet's row.\n"
)


class TableParser:
    """Extracts an optional ```table JSON block, validates its shape, strips it from the
    display text — same contract as answer.py's ChartParser. Malformed JSON is treated
    as no table, never shown broken."""

    def extract(self, text: str) -> tuple[str, dict | None]:
        match = TABLE_PATTERN.search(text)
        if not match:
            return text, None
        # Strip the fenced block whether or not it parsed — see ChartParser.extract's
        # matching comment; a malformed block is the model's ATTEMPT at a table, not
        # prose worth keeping visible.
        stripped = (text[: match.start()] + text[match.end() :]).strip()
        return stripped, self._validate(match.group(1))

    def _validate(self, raw: str) -> dict | None:
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        columns = data.get("columns")
        rows = data.get("rows")
        if not isinstance(columns, list) or not columns:
            return None
        if not isinstance(rows, list):
            return None
        cleaned_rows = []
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                return None
            cleaned_rows.append(["" if v is None else v for v in row])
        return {"columns": [str(c) for c in columns], "rows": cleaned_rows}


def extract_table(text: str) -> tuple[str, dict | None]:
    return TableParser().extract(text)


@dataclass
class DevicePoint:
    device_type: str  # "inlet" | "outlet"
    device_id: str
    pid: str
    location: str  # the inlet's location, or the outlet's sub-place name


@dataclass
class PlaceDevices:
    place_name: str
    ulb: str
    points: list[DevicePoint]


# Matches the export format in the CTC places/devices doc — see the script that
# generated it. A chunk that doesn't match this shape isn't a place-doc chunk at all.
_HEADER_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_ULB_RE = re.compile(r"^ULB:\s*(.+)$", re.MULTILINE)
_INLET_RE = re.compile(r"^Inlet:\s*device_id=([^,]+),\s*pid=([^,]+),\s*location=(.+)$", re.MULTILINE)
_OUTLET_RE = re.compile(r"^-\s*device_id=([^,]+),\s*pid=([^,]+),\s*sub_place=(.+)$", re.MULTILINE)


def parse_place_blocks(text: str) -> list[PlaceDevices]:
    """Parses every place block found in a chunk of the places/devices doc. A retrieved
    chunk commonly holds several places (the chunker doesn't align to place boundaries),
    not just one — callers must pick the right block(s) themselves (see
    find_matching_places) rather than assuming the first one is the answer."""
    headers = list(_HEADER_RE.finditer(text))
    blocks = []
    for i, h in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block_text = text[h.start() : end]
        ulb = _ULB_RE.search(block_text)
        inlet = _INLET_RE.search(block_text)
        if not ulb or not inlet:
            continue
        points = [DevicePoint("inlet", inlet.group(1).strip(), inlet.group(2).strip(), inlet.group(3).strip())]
        for m in _OUTLET_RE.finditer(block_text):
            points.append(DevicePoint("outlet", m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
        blocks.append(PlaceDevices(h.group(1).strip(), ulb.group(1).strip(), points))
    return blocks


_WORD_RE = re.compile(r"[a-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Common English function words carry no place-identifying signal, but count toward
# token-overlap scoring just like a real word unless filtered — confirmed to cause a
# real false match: "Road_6_to_Road_8" (a real BBSR sub-place) overlapped an unrelated
# question on "to" (boilerplate connector, plus leftover from date-phrase stripping)
# PLUS a coincidental "8" (from "Zone 08", nothing to do with the road's own numbering),
# crossing the match threshold together where neither alone would have.
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "in", "and", "or", "to", "at", "is", "on", "with",
    "from", "by", "this", "that", "it", "as", "be", "are", "was", "were",
})


def _tokens(text: str) -> set[str]:
    # Strip leading zeros so "zone 8" overlaps "Zone_08_..." — zero-padded numbering in
    # these place names is a formatting detail, not something a user typing a shorthand
    # reference is expected to reproduce. `or t` guards an all-zero token ("00") from
    # collapsing to an empty string.
    return {
        t.lstrip("0") or t
        for t in _WORD_RE.findall(text.lower())
        if t not in _STOPWORDS
    }


def _normalized(text: str) -> str:
    return _NON_ALNUM_RE.sub("", text.lower())


def find_matching_places(question: str, places: list[PlaceDevices]) -> list[PlaceDevices]:
    """Retrieval finds the right CHUNK(s) (which can each hold several places); this
    picks WHICH place(s) within them the question is actually about — supports asking
    about more than one place at once, returning every place with any real signal,
    best match first (deduplicated by name, since the same place can appear in more
    than one retrieved chunk). Empty list (never guesses) when nothing matches at all —
    the caller falls back to a normal answer.

    Two signals, deliberately different strength: normalized token overlap against the
    place name and every sub-place/outlet location name (e.g. "data for
    Zone_11_Gorakabar", "pressure at Deer_Park") — and a full-string substring match on
    a point's device id (e.g. "status of device 00-80-F4-2D-32-35"). A device id's
    individual hex fragments ("00", "35") are too short and common to score by token
    overlap without false-positiving on unrelated questions, so a whole-id substring
    match is scored as a strong, unambiguous hit instead.

    A lone overlapping token is usually noise — "zone" and "ESR" are generic words
    shared by nearly every place/point name in the CTC dataset — UNLESS that one token
    IS the entire PLACE name (e.g. "Acharyavihar", a real single-word BBSR/Puri place
    name): then every one of the name's tokens is present, so it's a full match, not a
    partial one, and gets the same strong-match exemption as a device id hit. This
    exemption deliberately does NOT extend to a sub-place/point LOCATION — "ESR" is
    itself a single-token location shared by nearly every place's inlet, so a bare
    question like "...ESR (inlet)..." would otherwise strong-match almost the entire
    corpus (confirmed: it did, matching 17 of ~20 places, before this was scoped to
    place names only)."""
    q_tokens = _tokens(question)
    q_normalized = _normalized(question)
    scored: dict[str, tuple[int, PlaceDevices]] = {}
    for place in places:
        place_tokens = _tokens(place.place_name)
        score = len(q_tokens & place_tokens)
        if place_tokens and place_tokens <= q_tokens:
            score = max(score, 10)
        for point in place.points:
            loc_tokens = _tokens(point.location)
            score = max(score, len(q_tokens & loc_tokens))
            device_id_normalized = _normalized(point.device_id)
            if device_id_normalized and device_id_normalized in q_normalized:
                score = max(score, 10)
        if score < 2:
            continue
        existing = scored.get(place.place_name)
        if existing is None or score > existing[0]:
            scored[place.place_name] = (score, place)
    return [place for _score, place in sorted(scored.values(), key=lambda sp: sp[0], reverse=True)]


def redis_prefix_for_ulb(ulb: str) -> str | None:
    return ULB_REDIS_PREFIX.get(ulb.strip().lower())


def _space_redis_connector(space_id: str) -> dict | None:
    """The first Redis connector configured for this space, or None — callers treat
    that the same as "no live data source configured", not an error."""
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT host, port, username, encrypted_password, db_index, tls "
            "FROM connectors WHERE space_id=? AND kind='redis' ORDER BY created_at LIMIT 1",
            (space_id,),
        ).fetchone()
    return dict(row) if row else None


# {notification} is a Redis Cluster hash-tag (the braces are literal) — it forces every
# key for a device onto the same cluster slot. Not optional; a key built without it
# doesn't exist. Confirmed against real production keys.
def _redis_key(prefix: str, point: DevicePoint) -> str:
    return f"{{notification}}:{prefix}:device:latest:{point.device_type}:{point.device_id}:{point.pid}"


def fetch_live_readings(space_id: str, ulb: str, points: list[DevicePoint]) -> dict[str, dict | None] | None:
    """One MGET for every point's key, keyed by that key. Returns None (not an error)
    when there's no known Redis prefix for this ULB or no Redis connector configured
    for the space — both mean "can't answer this live" upstream. An individual point's
    value is None when its key is missing or unparseable, so one dead sensor doesn't
    sink the whole place's answer."""
    prefix = redis_prefix_for_ulb(ulb)
    if prefix is None:
        return None
    connector = _space_redis_connector(space_id)
    if connector is None:
        return None
    try:
        password = decrypt(connector["encrypted_password"], settings.connector_encryption_key)
    except RuntimeError:
        return None

    keys = [_redis_key(prefix, p) for p in points]
    client = None
    try:
        client = redis.Redis(
            host=connector["host"], port=connector["port"], username=connector["username"] or None,
            password=password or None, db=connector["db_index"] or 0, ssl=bool(connector["tls"]),
            socket_connect_timeout=5, socket_timeout=5, protocol=2, decode_responses=True,
        )
        raw_values = client.mget(keys)
    except Exception:
        return None
    finally:
        if client is not None:
            client.close()

    result: dict[str, dict | None] = {}
    for key, raw in zip(keys, raw_values, strict=True):
        try:
            result[key] = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            result[key] = None
    return result


def _first(values: object) -> object:
    """payload.pressure/flow/totalizer are single-element arrays in the real payload —
    defensive about that shape rather than assuming it, since a malformed/empty array
    or a non-list value should read as "no reading", not raise."""
    if isinstance(values, list):
        return values[0] if values else None
    return values


def build_live_data_context(places: list[PlaceDevices], readings: dict[str, dict | None]) -> str:
    """The already-extracted, human-readable context handed to the generation LLM, one
    section per matched place — supports the multi-place case (see
    find_matching_places), not just a single one. Each Redis key holds exactly one
    sub-place's own reading (no shared multi-sub-place blob, no field-number selection
    needed) — read straight out of that sub-place's payload. Includes device id/pid here
    regardless — LIVE_DATA_SYSTEM_PROMPT is what tells the model not to surface them
    unless asked; the model still needs them internally in case it is asked. Label
    format ("<location> (<type>)") matches what the prompt asks the model's own output
    to look like, so the model has a consistent shape to carry through rather than
    inventing its own."""
    sections = []
    for place in places:
        prefix = redis_prefix_for_ulb(place.ulb)
        lines = [f"Place: {place.place_name} (ULB: {place.ulb})"]
        chlorine = None  # same value in every outlet payload for this place — report once, not per sub-place
        for p in place.points:
            reading = readings.get(_redis_key(prefix, p)) if prefix else None
            payload = (reading or {}).get("payload") if reading else None
            label = f"{p.location} ({p.device_type}) (device_id {p.device_id}, pid {p.pid})"
            if not payload:
                lines.append(f"- {label}: reading unavailable")
                continue
            pressure = _first(payload.get("pressure"))
            flow = _first(payload.get("flow"))
            totalizer = _first(payload.get("totalizer"))
            reading_line = f"- {label}: pressure={pressure}, flow={flow}, totalizer={totalizer}"
            if p.device_type == "inlet" and payload.get("level") is not None:
                reading_line += f", level={payload['level']}"
            lines.append(reading_line)
            if chlorine is None and p.device_type == "outlet":
                chlorine = _first(payload.get("chlorine"))
        if chlorine is not None:
            lines.append(f"Chlorine (overall for this place, same across every outlet): {chlorine}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
