"""Document loading, chunking, and lightweight grounded retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from aqb.models import NoteChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_text_from_path(path_or_text: str) -> tuple[str, str]:
    """Load text from a file path if it exists, otherwise treat input as raw text.

    Returns:
        Tuple of (content, source_label).
    """
    candidate = Path(path_or_text.strip())
    if not candidate.exists() or not candidate.is_file():
        return path_or_text, "inline-input"

    suffix = candidate.suffix.lower()
    if suffix in {".txt", ".md"}:
        return candidate.read_text(encoding="utf-8", errors="replace"), candidate.name

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to read PDF files. Install with: pip install pymupdf"
            ) from exc
        pages: list[str] = []
        with fitz.open(candidate) as doc:
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
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs), candidate.name

    return candidate.read_text(encoding="utf-8", errors="replace"), candidate.name


def chunk_notes(
    text: str,
    source: str,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[NoteChunk]:
    """Split study notes into overlapping chunks with deterministic chunk IDs."""
    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[NoteChunk] = []
    start = 0
    idx = 1
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                NoteChunk(
                    chunk_id=f"{source}#C{idx}",
                    source=source,
                    text=piece,
                )
            )
            idx += 1
        if end >= len(cleaned):
            break
        start += step
    return chunks


class NoteRetriever:
    """Zero-dependency TF-IDF retriever for finding relevant note chunks per batch."""

    def __init__(self, chunks: list[NoteChunk]) -> None:
        self.chunks = chunks
        self._chunk_tokens = [_tokenize(c.text) for c in chunks]
        self._doc_freq: Counter[str] = Counter()
        for tokens in self._chunk_tokens:
            self._doc_freq.update(set(tokens))
        self._num_docs = max(1, len(chunks))

    def retrieve(self, queries: list[str], top_k: int = 8) -> list[NoteChunk]:
        """Return top-k note chunks relevant to the combined batch queries."""
        if not self.chunks:
            return []
        query_tokens = _tokenize(" ".join(queries))
        if not query_tokens:
            return self.chunks[:top_k]

        scored: list[tuple[float, int]] = []
        for idx, doc_tokens in enumerate(self._chunk_tokens):
            if not doc_tokens:
                continue
            tf = Counter(doc_tokens)
            score = 0.0
            for qt in query_tokens:
                if qt in tf:
                    idf = math.log((self._num_docs + 1) / (self._doc_freq[qt] + 1)) + 1.0
                    score += tf[qt] * idf
            scored.append((score, idx))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_indices = [idx for score, idx in scored[:top_k] if score > 0]
        if not top_indices:
            return self.chunks[:top_k]
        return [self.chunks[i] for i in top_indices]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
