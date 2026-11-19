"""Document loading, chunking, and lightweight grounded retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from aqb.models import NoteChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PAGE_RE = re.compile(r"\[Page (\d+)\]")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".pptx"}

# Common English + exam-phrasing stopwords that carry no retrieval signal.
_STOPWORDS = frozenset(
    """a an the and or of to in on for with by is are was were be been being as at from that this
    these those it its into than then what which who whom how why when where explain define describe
    discuss write short note notes brief briefly detail detailed list state give mention compare
    differentiate between with neat diagram example examples marks mark any two three four five
    suitable illustrate elaborate outline""".split()
)


def load_text_from_path(path_or_text: str) -> tuple[str, str]:
    """Load text from a file path if it exists, otherwise treat input as raw text.

    Returns:
        Tuple of (content, source_label).
    """
    stripped = path_or_text.strip()
    candidate = Path(stripped).expanduser() if stripped and "\n" not in stripped else None
    if candidate is None or not candidate.is_file():
        return path_or_text, "inline-input"

    suffix = candidate.suffix.lower()
    if suffix in {".txt", ".md"}:
        return candidate.read_text(encoding="utf-8", errors="replace"), candidate.name

    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to read PDF files. Install with: pip install pymupdf"
            ) from exc
        pages: list[str] = []
        with pymupdf.open(candidate) as doc:
            for idx, page in enumerate(doc, start=1):
                pages.append(f"[Page {idx}]\n{page.get_text('text')}")
        return "\n\n".join(pages), candidate.name

    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError as exc:
            raise RuntimeError(
                "python-docx is required to read DOCX files. Install with: pip install python-docx"
            ) from exc
        document = docx.Document(str(candidate))
        blocks = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))
        return "\n\n".join(blocks), candidate.name

    if suffix == ".pptx":
        try:
            from pptx import Presentation  # python-pptx
        except ImportError as exc:
            raise RuntimeError(
                "python-pptx is required to read PPTX files. Install with: pip install python-pptx"
            ) from exc
        slides: list[str] = []
        for idx, slide in enumerate(Presentation(str(candidate)).slides, start=1):
            texts = [
                shape.text_frame.text
                for shape in slide.shapes
                if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip()
            ]
            slides.append(f"[Page {idx}]\n" + "\n".join(texts))
        return "\n\n".join(slides), candidate.name

    return candidate.read_text(encoding="utf-8", errors="replace"), candidate.name


def load_many(inputs: list[str]) -> list[tuple[str, str]]:
    """Load several files/raw-text inputs, de-duplicating source labels."""
    loaded: list[tuple[str, str]] = []
    seen: Counter[str] = Counter()
    for item in inputs:
        if not item or not item.strip():
            continue
        text, source = load_text_from_path(item)
        seen[source] += 1
        if seen[source] > 1:
            source = f"{source}({seen[source]})"
        loaded.append((text, source))
    return loaded


def chunk_notes(
    text: str,
    source: str,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[NoteChunk]:
    """Split notes into overlapping chunks, preferring paragraph/sentence boundaries.

    Chunk IDs include the page number when the text carries `[Page N]` markers
    (PDF/PPTX), e.g. `notes.pdf#p4.C7`, so citations are human-checkable.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    page_marks = [(m.start(), int(m.group(1))) for m in _PAGE_RE.finditer(cleaned)]

    def page_at(pos: int) -> int | None:
        page = None
        for mark_pos, num in page_marks:
            if mark_pos <= pos:
                page = num
            else:
                break
        return page

    chunks: list[NoteChunk] = []
    start = 0
    idx = 1
    n = len(cleaned)
    while start < n:
        end = min(n, start + chunk_size)
        if end < n:
            window_start = start + int(chunk_size * 0.6)
            for sep in ("\n\n", "\n", ". "):
                cut = cleaned.rfind(sep, window_start, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = cleaned[start:end].strip()
        if piece:
            # Use the page where most of the chunk's content lives (first marker inside, else last before).
            page = page_at(start)
            inner = _PAGE_RE.search(piece)
            if inner and inner.start() < len(piece) // 2:
                page = int(inner.group(1))
            label = f"p{page}.C{idx}" if page is not None else f"C{idx}"
            chunks.append(NoteChunk(chunk_id=f"{source}#{label}", source=source, text=piece))
            idx += 1
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return chunks


class NoteRetriever:
    """Zero-dependency BM25 retriever for finding relevant note chunks per question."""

    def __init__(self, chunks: list[NoteChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tfs = [Counter(tokenize(c.text)) for c in chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 1.0
        self._df: Counter[str] = Counter()
        for tf in self._tfs:
            self._df.update(tf.keys())
        self._n = max(1, len(chunks))

    def total_chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[tuple[float, int]]:
        q_terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for idx, tf in enumerate(self._tfs):
            if not tf:
                continue
            length_norm = self.k1 * (1 - self.b + self.b * self._lens[idx] / self._avg_len)
            s = 0.0
            for term in q_terms:
                f = tf.get(term)
                if f:
                    s += self._idf(term) * (f * (self.k1 + 1)) / (f + length_norm)
            if s > 0:
                scored.append((s, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def retrieve(self, queries: list[str], top_k: int = 8) -> list[NoteChunk]:
        """Return top-k chunks for the combined queries (kept for backwards compatibility)."""
        if not self.chunks:
            return []
        top = [idx for _, idx in self.score(" ".join(queries))[:top_k]]
        return [self.chunks[i] for i in top] if top else self.chunks[:top_k]

    def retrieve_for_batch(
        self,
        queries: list[str],
        per_query: int = 3,
        max_chunks: int = 14,
    ) -> list[NoteChunk]:
        """Retrieve top chunks *per question* and union them (round-robin), in document order.

        Per-question retrieval guarantees every question in the batch gets its own
        supporting evidence, instead of one dominant topic crowding out the rest.
        """
        if not self.chunks:
            return []
        ranked = [[idx for _, idx in self.score(q)[:per_query]] for q in queries]
        selected: list[int] = []
        for rank in range(per_query):
            for hits in ranked:
                if rank < len(hits) and hits[rank] not in selected:
                    selected.append(hits[rank])
                if len(selected) >= max_chunks:
                    break
            if len(selected) >= max_chunks:
                break
        if not selected:
            return self.chunks[:max_chunks]
        return [self.chunks[i] for i in sorted(selected)]


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]
