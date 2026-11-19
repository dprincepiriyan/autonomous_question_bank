"""Question-bank parsing (LLM + deterministic heuristic) and topic-importance analysis."""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

from aqb.models import Question, TopicImportance

log = logging.getLogger(__name__)

VALID_MARKS = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 20}
PART_DEFAULT_MARKS = {"a": 2, "b": 13, "c": 15}

_PART_RE = re.compile(r"^\s*part\s*[-–—:]?\s*([a-c])\b", re.I)
_UNIT_RE = re.compile(
    r"^\s*(unit|module|chapter|lesson)\s*[-–—:]?\s*([ivxlc]+|\d+)\b[\s:.\-–—]*(.*)$", re.I
)
_MARK_HEADER_RE = re.compile(
    r"^\s*\(?\s*(\d{1,2})\s*[- ]?\s*marks?\b|\b\d{1,2}\s*[x×*]\s*(\d{1,2})\s*=", re.I
)
_NUM_RE = re.compile(r"^\s*(?:q(?:uestion)?\s*\.?\s*)?(\d{1,3})\s*[\.\):\-]\s*(\S.*)$", re.I)
_BULLET_RE = re.compile(r"^\s*[-•*▪●]\s+(\S.*)$")
_OR_RE = re.compile(r"^\s*\(?\s*or\s*\)?\s*$", re.I)
_ALT_RE = re.compile(r"^\s*\(?([ab])\)\s+(\S.*)$", re.I)
_TAG_MARKS_RE = re.compile(r"[\[\(]\s*(\d{1,2})\s*(?:m|marks?)\s*[\]\)]", re.I)
_TRAILING_MARKS_RE = re.compile(
    r"\((\d{1,2})\)(?=\s*(?:$|\(?[ivx]+\)|\(?co\s*\d|\(?bt|\(?k\d|\[|\())", re.I
)
_TOPIC_TAG_RE = re.compile(r"\(?\s*topic\s*:\s*([^\]\)\n]+)\)?", re.I)


# --------------------------------------------------------------------------------------
# Heuristic (deterministic) parser
# --------------------------------------------------------------------------------------
def _header_marks(line: str) -> int | None:
    m = _MARK_HEADER_RE.search(line)
    if not m:
        return None
    value = int(m.group(2) or m.group(1))
    return value if value in VALID_MARKS else None


def _inline_marks(text: str) -> int | None:
    tags = [int(v) for v in _TAG_MARKS_RE.findall(text)]
    if tags:
        total = sum(tags)
        return total if total in VALID_MARKS or len(tags) > 1 else None
    trailing = [int(v) for v in _TRAILING_MARKS_RE.findall(text)]
    if trailing:
        total = sum(trailing)
        if total in VALID_MARKS:
            return total
    return None


def _is_header(line: str) -> bool:
    if _PART_RE.match(line) or _UNIT_RE.match(line):
        return True
    if re.match(r"^\s*\(?\s*\d{1,2}\s*[- ]?\s*marks?\b", line, re.I) and len(line) < 60 and "?" not in line:
        return True
    # Short "2 Marks" / "Part B (5 x 13 = 65 marks)" style lines without a question.
    return bool(_header_marks(line)) and len(line) < 80 and "?" not in line and not _NUM_RE.match(line)


def heuristic_parse_questions(raw: str) -> list[Question]:
    """Deterministically parse common question-bank layouts.

    Handles: PART A/B/C sections, "N x M = T" headers, "2 Marks" headers, UNIT headings,
    numbered questions ("1.", "Q1)", "12 -"), bullet lists, multi-line questions,
    (OR) alternatives (11(a)/11(b)), inline mark tags ("[2M]", "(13 marks)", "(7) ... (6)").
    """
    lines = [ln.rstrip() for ln in raw.splitlines()]
    has_numbering = any(_NUM_RE.match(ln) for ln in lines if not _is_header(ln))

    section_marks = 2
    topic = "General"
    items: list[dict] = []
    current: dict | None = None
    force_new = False

    def flush() -> None:
        nonlocal current
        if current and current["text"].strip():
            items.append(current)
        current = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if _OR_RE.match(stripped):
            flush()
            force_new = True
            continue

        unit = _UNIT_RE.match(stripped)
        if unit and not _NUM_RE.match(stripped):
            flush()
            title = unit.group(3).strip(" :-–—")
            topic = title or f"{unit.group(1).title()} {unit.group(2).upper()}"
            continue

        part = _PART_RE.match(stripped)
        if part or _is_header(stripped):
            flush()
            hm = _header_marks(stripped)
            if hm:
                section_marks = hm
            elif part:
                section_marks = PART_DEFAULT_MARKS[part.group(1).lower()]
            continue

        num = _NUM_RE.match(stripped)
        bullet = _BULLET_RE.match(stripped)
        alt = _ALT_RE.match(stripped)
        starts_new = (
            bool(num)
            or (bullet is not None and not has_numbering)
            or (not has_numbering and not bullet)
            or (force_new and current is None)
            or (alt is not None and alt.group(1).lower() == "b" and current is not None and current["marks"] > 5)
        )

        if current is None and has_numbering and not num and not force_new and alt is None:
            continue  # titles / instructions before the first numbered question of a section

        if starts_new or current is None:
            flush()
            text = num.group(2) if num else (bullet.group(1) if bullet else stripped)
            current = {"text": text, "marks": section_marks, "topic": topic}
            force_new = False
        else:
            current["text"] += " " + stripped

    flush()

    questions: list[Question] = []
    for idx, item in enumerate(items, start=1):
        text = re.sub(r"\s+", " ", item["text"]).strip()
        marks = _inline_marks(text) or item["marks"]
        q_topic = item["topic"]
        tag = _TOPIC_TAG_RE.search(text)
        if tag:
            q_topic = tag.group(1).strip()
        questions.append(Question(id=f"Q{idx}", text=text, marks=marks, topic=q_topic))
    return questions


# --------------------------------------------------------------------------------------
# Splitting for LLM parsing of large banks
# --------------------------------------------------------------------------------------
def split_question_bank(raw: str, max_chars: int = 6000) -> list[tuple[str, str]]:
    """Split a bank into line-aligned slices, returning (slice_text, last_header_before_slice).

    The header context lets the LLM keep correct mark values / units across slices.
    Slices never cut through a numbered question.
    """
    lines = raw.splitlines()
    slices: list[tuple[str, str]] = []
    buf: list[str] = []
    size = 0
    last_header = ""
    header_for_slice = ""

    for line in lines:
        stripped = line.strip()
        is_boundary = bool(_NUM_RE.match(stripped)) or _is_header(stripped) or not stripped
        if size + len(line) > max_chars and buf and is_boundary:
            slices.append(("\n".join(buf), header_for_slice))
            buf, size = [], 0
            header_for_slice = last_header
        buf.append(line)
        size += len(line) + 1
        if stripped and (_is_header(stripped) or _UNIT_RE.match(stripped)):
            last_header = stripped
    if buf and "\n".join(buf).strip():
        slices.append(("\n".join(buf), header_for_slice))
    return slices


def renumber(questions: list[Question]) -> list[Question]:
    for idx, q in enumerate(questions, start=1):
        q.id = f"Q{idx}"
    return questions


# --------------------------------------------------------------------------------------
# Topic importance (deterministic) + repeated questions + key concepts
# --------------------------------------------------------------------------------------
def compute_topic_importance(questions: list[Question]) -> list[TopicImportance]:
    """Rank topics by total mark weightage. Numbers are computed, never LLM-generated."""
    groups: dict[str, list[Question]] = defaultdict(list)
    canonical: dict[str, str] = {}
    for q in questions:
        key = q.topic.strip().casefold()
        canonical.setdefault(key, q.topic.strip() or "General")
        groups[key].append(q)

    grand_total = sum(q.marks for q in questions) or 1
    ranked = sorted(
        groups.items(),
        key=lambda kv: (sum(q.marks for q in kv[1]), len(kv[1])),
        reverse=True,
    )
    results: list[TopicImportance] = []
    for rank, (key, qs) in enumerate(ranked):
        total = sum(q.marks for q in qs)
        share = total / grand_total
        long_count = sum(1 for q in qs if q.marks > 5)
        if rank == 0 or share >= 0.20:
            priority = "High"
        elif share >= 0.08 or long_count >= 2:
            priority = "Medium"
        else:
            priority = "Low"
        results.append(
            TopicImportance(
                topic=canonical[key],
                question_count=len(qs),
                total_marks=total,
                priority=priority,
                rationale=(
                    f"{share:.0%} of total marks; {long_count} long-answer and "
                    f"{len(qs) - long_count} short-answer question(s)."
                ),
            )
        )
    return results


_CONCEPT_STOP = frozenset(
    """a an the and or of to in on for with by is are was were be as at from that this these those it its
    into what which who how why when where explain define describe discuss write short note notes brief
    briefly detail detailed list state give mention compare differentiate between neat diagram example
    examples marks mark any two three four five suitable illustrate elaborate outline various different
    types important their your you can using use used about also do does q part unit meant means mean
    repeated previous year years asked""".split()
)


def _concept_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9\-]+", text.lower()) if t not in _CONCEPT_STOP]


def key_concepts(questions: list[Question], top_n: int = 15) -> list[dict]:
    """Most frequently examined terms/phrases, weighted by marks (bigrams preferred)."""
    weights: Counter[str] = Counter()
    hits: dict[str, set[str]] = defaultdict(set)
    for q in questions:
        toks = _concept_tokens(q.text)
        grams = set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}
        for g in grams:
            weights[g] += q.marks * (1.6 if " " in g else 1.0)
            hits[g].add(q.id)

    # Keep terms asked in 2+ questions; drop unigrams subsumed by an equally frequent bigram.
    candidates = [(g, w) for g, w in weights.most_common() if len(hits[g]) >= 2]
    chosen: list[dict] = []
    for g, w in candidates:
        if " " not in g and any(
            g in c["term"].split() and c["questions"] >= len(hits[g]) for c in chosen
        ):
            continue
        chosen.append({"term": g, "weight": round(w, 1), "questions": len(hits[g])})
        if len(chosen) >= top_n:
            break
    return chosen


def find_repeated_questions(questions: list[Question], threshold: float = 0.7) -> list[list[str]]:
    """Group near-duplicate questions (Jaccard similarity on content words). Repeats = likely exam picks."""
    token_sets = {q.id: set(_concept_tokens(q.text)) for q in questions}
    parent = {q.id: q.id for q in questions}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = [q.id for q in questions]
    for i, a in enumerate(ids):
        sa = token_sets[a]
        if not sa:
            continue
        for b in ids[i + 1 :]:
            sb = token_sets[b]
            if not sb:
                continue
            if sa == sb or (min(len(sa), len(sb)) >= 2 and len(sa & sb) / len(sa | sb) >= threshold):
                parent[find(b)] = find(a)

    groups: dict[str, list[str]] = defaultdict(list)
    for qid in ids:
        groups[find(qid)].append(qid)
    return [g for g in groups.values() if len(g) > 1]


# --------------------------------------------------------------------------------------
# Sub-topic assignment from the notes' own headings (deterministic, no LLM needed)
# --------------------------------------------------------------------------------------
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.!?]{2,70})$")


def _clean_heading(text: str) -> str:
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text.strip())
    text = re.sub(r"\s*\(.*?\)\s*$", "", text)
    return text.strip(" :-–—*").strip() or text.strip()


def split_notes_sections(text: str) -> list[tuple[str, str]]:
    """Split notes into (heading, body) sections at the most useful heading level.

    The chosen level is the shallowest one with at least 3 headings, so sections are
    lesson sub-topics (e.g. "CPU Scheduling") rather than tiny sub-sub-headings.
    """
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    for i, line in enumerate(lines):
        s = line.strip()
        md = _MD_HEADING_RE.match(s)
        if md:
            headings.append((i, len(md.group(1)), _clean_heading(md.group(2))))
            continue
        num = _NUMBERED_HEADING_RE.match(s)
        if num and len(s) <= 75:
            headings.append((i, num.group(1).count(".") + 1, _clean_heading(num.group(2))))
        elif s.isupper() and 3 <= len(s) <= 60 and len(s.split()) <= 8 and not s.startswith("[PAGE"):
            headings.append((i, 1, s.title()))

    if not headings:
        return []
    levels = sorted({lvl for _, lvl, _ in headings})
    chosen = next((lvl for lvl in levels if sum(1 for _, l, _ in headings if l == lvl) >= 3), levels[-1])
    marks = [(i, title) for i, lvl, title in headings if lvl == chosen]

    sections = []
    for n, (start, title) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start + 1 : end])
        if body.strip():
            sections.append((title, body))
    return sections


def assign_topics_from_notes(questions: list[Question], notes_texts: list[str]) -> bool:
    """Tag each question with the best-matching notes section heading. Returns True if applied."""
    from aqb.ingestion import NoteRetriever
    from aqb.models import NoteChunk

    sections: list[tuple[str, str]] = []
    for text in notes_texts:
        sections.extend(split_notes_sections(text))
    if len(sections) < 2:
        return False

    # Heading words are repeated to weight them above body text.
    chunks = [
        NoteChunk(chunk_id=str(i), source="sections", text=f"{(title + ' ') * 3}\n{body}")
        for i, (title, body) in enumerate(sections)
    ]
    retriever = NoteRetriever(chunks)
    changed = False
    for q in questions:
        scored = retriever.score(q.text)
        if scored:
            q.topic = sections[scored[0][1]][0]
            changed = True
    return changed


def needs_topic_refinement(questions: list[Question]) -> bool:
    """True when parsing produced too few distinct topics to be useful for prioritisation."""
    if len(questions) < 6:
        return False
    distinct = {q.topic.casefold() for q in questions}
    return len(distinct) <= 1 or (len(distinct) <= 2 and len(questions) >= 20)
