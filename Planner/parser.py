"""Step 2 - split syllabus text into units and topics.

Works on the usual university layouts:

    Unit I: Introduction to Machine Learning (08 Hrs)
    Definition, types of learning, applications, ...

    UNIT-2 Supervised Learning - Linear regression, logistic regression; SVM
    Module 3: ...

Headings are found with regular expressions; topics are split on commas,
semicolons, colons, bullets and sentence ends (commas inside brackets are kept).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

from .models import Topic, Unit

UNIT_RE = re.compile(
    r"^\s*(?:unit|module|chapter|section)\s*[-–—:.#]?\s*(?P<num>\d+|[ivxlcdm]+)\b\s*[-–—:.)]*\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
NUMBERED_RE = re.compile(r"^\s*(?P<num>\d{1,2})[.)]\s+(?P<rest>[A-Z].*)$")
STOP_RE = re.compile(
    r"^\s*(course\s+outcomes?|course\s+objectives?|learning\s+outcomes?|text\s*books?|"
    r"reference\s*books?|references?|suggested\s+readings?|e-?resources|web\s+resources|"
    r"list\s+of\s+(experiments|practicals|assignments)|prerequisites?|pre-?requisites?|"
    r"assessment|evaluation\s+scheme|examination\s+scheme|co-?po\s+mapping|mapping)\b",
    re.IGNORECASE,
)
HOURS_RE = re.compile(r"[\(\[]?\s*(\d{1,3})\s*(?:hrs?\.?|hours?|lectures?|periods?)\s*[\)\]]?", re.IGNORECASE)
PAGE_NO_RE = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE)
BULLET_START_RE = re.compile(r"^\s*([•●▪■◦◆►➢\-*]|\d+[.)]|\(?[a-z][.)])\s+")
BULLET_CHARS_RE = re.compile(r"[•●▪■◦◆►➢✓]")
ABBREVIATIONS = ("e.g", "i.e", "vs", "etc", "fig", "eq", "approx", "cf")


# --------------------------------------------------------------------------- helpers
def _roman_to_int(token: str) -> int:
    token = token.lower()
    if token.isdigit():
        return int(token)
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total, prev = 0, 0
    for ch in reversed(token):
        v = values.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total or 1


def _prepare_lines(text: str) -> List[str]:
    text = text.replace("\r", "\n").replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join words hyphenated at line end
    lines = [re.sub(r"[ ]{2,}", " ", ln).strip() for ln in text.split("\n")]

    # drop page numbers and repeated headers/footers (same short line on 3+ pages)
    counts = Counter(ln for ln in lines if ln)
    cleaned = []
    for ln in lines:
        if PAGE_NO_RE.match(ln):
            continue
        if ln and counts[ln] >= 3 and len(ln) < 80 and not UNIT_RE.match(ln):
            continue
        cleaned.append(ln)
    return cleaned


def _split_heading_rest(rest: str) -> Tuple[str, str]:
    """'Intro to ML: definition, types' -> ('Intro to ML', 'definition, types')."""
    rest = rest.strip(" :-–—.")
    if not rest:
        return "", ""
    parts = re.split(r"\s[-–—]\s|:\s", rest, maxsplit=1)
    if len(parts) == 2 and parts[0].strip():
        return parts[0].strip(), parts[1].strip()
    if len(rest) > 70 or rest.count(",") >= 2 or ";" in rest:
        return "", rest
    return rest, ""


def _is_wrapped(prev: str) -> bool:
    """True if `prev` looks like a line that continues on the next line."""
    if prev.endswith(","):
        return True
    return len(prev) >= 60 and not prev.endswith((".", ";", ":"))


def _join_lines(lines: List[str]) -> List[str]:
    chunks, buf = [], ""
    for ln in lines:
        if not ln:
            if buf:
                chunks.append(buf)
                buf = ""
            continue
        if buf and not BULLET_START_RE.match(ln) and _is_wrapped(buf):
            buf += " " + ln
        else:
            if buf:
                chunks.append(buf)
            buf = ln
    if buf:
        chunks.append(buf)
    return chunks


def _split_top_level(s: str, seps: str = ",;:") -> List[str]:
    """Split on separators that are not inside (), [] or {}."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch in seps and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _clean_topic(raw: str) -> Optional[str]:
    t = raw.replace("§", ".")
    t = BULLET_START_RE.sub("", t)
    t = re.sub(r"^\s*(and|or|&|e\.g\.?|i\.e\.?|including|such as)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" .,-–—:;|")
    t = re.sub(r"\b(etc|and so on|and more)\b\.?$", "", t, flags=re.IGNORECASE).strip(" .,-–—:;")
    if len(t) < 3 or t.isdigit() or not re.search(r"[A-Za-z]", t):
        return None
    if len(t) > 160:
        t = t[:157].rstrip() + "..."
    if t[0].islower() and not (len(t) > 1 and t[1].isupper()):   # keep names like "kNN"
        t = t[0].upper() + t[1:]
    return t


def split_topics(chunk: str) -> List[str]:
    """Break one block of syllabus text into individual topic names."""
    for ab in ABBREVIATIONS:                                   # protect "e.g." etc. from the full-stop split
        chunk = re.sub(rf"\b{re.escape(ab)}\.", ab.replace(".", "§") + "§", chunk, flags=re.IGNORECASE)
    chunk = BULLET_CHARS_RE.sub(";", chunk)
    chunk = re.sub(r"\s[-–—]\s", ";", chunk)                   # spaced dashes
    chunk = re.sub(r"\.(?=\s|$)", ";", chunk)                  # sentence ends
    topics = []
    for piece in _split_top_level(chunk):
        cleaned = _clean_topic(piece)
        if cleaned:
            topics.append(cleaned)
    return topics


# --------------------------------------------------------------------------- main API
def parse_syllabus(text: str) -> List[Unit]:
    """Return a list of `Unit` objects (each with its `Topic`s)."""
    lines = _prepare_lines(text)
    has_unit_headings = any(UNIT_RE.match(ln) for ln in lines)
    heading_re = UNIT_RE if has_unit_headings else NUMBERED_RE
    has_headings = has_unit_headings or any(NUMBERED_RE.match(ln) for ln in lines)

    if not has_headings:                                       # no structure at all: one big unit
        unit = Unit(number=1, title="Syllabus")
        _fill_unit(unit, [], "\n".join(lines))
        return [unit] if unit.topics else []

    units: dict = {}
    raw_lines: dict = {}
    order: List[int] = []
    current: Optional[int] = None
    active = False

    for ln in lines:
        m = heading_re.match(ln)
        if m and len(ln) < 400:
            num = _roman_to_int(m.group("num"))
            rest = m.group("rest") or ""
            hours = None
            hm = HOURS_RE.search(rest)
            if hm:
                hours = float(hm.group(1))
                rest = HOURS_RE.sub(" ", rest)
            title, inline = _split_heading_rest(rest)
            if num not in units:
                units[num] = Unit(number=num, title=title or f"Unit {num}", hours_hint=hours)
                raw_lines[num] = []
                order.append(num)
            else:                                              # same unit mentioned again -> keep first title
                if hours and not units[num].hours_hint:
                    units[num].hours_hint = hours
                if title and units[num].title.startswith("Unit "):
                    units[num].title = title
            current, active = num, True
            if inline:
                raw_lines[num].append(inline)
            continue

        if STOP_RE.match(ln) and len(ln) < 60:
            active = False
            continue
        if active and current is not None:
            raw_lines[current].append(ln)

    result = []
    for num in order:
        unit = units[num]
        _fill_unit(unit, raw_lines[num])
        if not unit.topics and not unit.title.startswith("Unit "):
            unit.topics = [Topic(unit.title, unit.number, unit.title)]   # heading was the only content
        if unit.topics:
            result.append(unit)
    return result


def _fill_unit(unit: Unit, lines: List[str], blob: str = "") -> None:
    chunks = _join_lines(lines if lines else blob.split("\n"))
    seen, topics = set(), []
    for chunk in chunks:
        for name in split_topics(chunk):
            key = name.lower()
            if key not in seen:
                seen.add(key)
                topics.append(Topic(name=name, unit_no=unit.number, unit_title=unit.title))
    unit.topics = topics
