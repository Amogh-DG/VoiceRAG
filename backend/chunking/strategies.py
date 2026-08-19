"""
backend/chunking/strategies.py
--------------------------------
Three concrete IChunker implementations, all producing TextChunk objects
that satisfy the stable cross-team schema:

    { text: str, document_id: str, metadata: dict }

Minimum required metadata keys (enforced by IChunker contract):
    chunk_strategy   – one of "fixed_overlap" | "sentence" | "metadata_aware"
    chunk_index      – zero-based position within the document

Strategy selection
──────────────────
Instantiate the class you want directly, or use ``get_chunker(name, **kwargs)``
to select by string name (used by the CLI).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.common.interfaces import IChunker
from backend.common.schemas import Document, TextChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Indic full-stop characters used as sentence boundaries
_INDIC_STOPS = "।॥"

# Regex that matches the END of a sentence — a stop character optionally
# followed by a closing quote/bracket, then a space or end-of-string.
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?%s])(?:\s*[\"'\)\]»])?(?=\s+|\Z)" % re.escape(_INDIC_STOPS)
)

# Paragraph break (two or more newlines)
_PARA_BREAK_RE = re.compile(r"\n{2,}")


def _make_chunk(
    text: str,
    document_id: str,
    index: int,
    strategy: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> TextChunk:
    """Construct a TextChunk with the mandatory metadata fields."""
    meta: Dict[str, Any] = {
        "chunk_strategy": strategy,
        "chunk_index": index,
    }
    if extra_meta:
        meta.update(extra_meta)
    return TextChunk(text=text, document_id=document_id, metadata=meta)


# ---------------------------------------------------------------------------
# Strategy 1 – Fixed-size + overlap
# ---------------------------------------------------------------------------

class FixedOverlapChunker(IChunker):
    """
    Splits document text into fixed-size windows with a configurable overlap.

    This is the simplest baseline strategy. Characters (not tokens) are counted,
    which keeps the implementation dependency-free and language-agnostic.

    Args:
        chunk_size:    Maximum characters per chunk (default 500).
        overlap:       Characters of overlap between adjacent chunks (default 50).
    """

    STRATEGY_NAME = "fixed_overlap"

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[TextChunk]:
        text = document.content
        if not text.strip():
            return []

        chunks: List[TextChunk] = []
        start = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(
                    _make_chunk(
                        text=chunk_text,
                        document_id=document.id,
                        index=len(chunks),
                        strategy=self.STRATEGY_NAME,
                        extra_meta={
                            "chunk_size": self.chunk_size,
                            "overlap": self.overlap,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
            if end == len(text):
                break
            start += step

        return chunks


# ---------------------------------------------------------------------------
# Strategy 2 – Sentence / semantic splitting
# ---------------------------------------------------------------------------

class SentenceChunker(IChunker):
    """
    Accumulates sentences until a soft max-character limit is reached, then
    starts a new chunk.  Sentence boundaries are detected via a regex that
    handles ASCII punctuation (.!?) and Indic full-stops (।॥), making it
    suitable for both English and Indic-language content.

    Args:
        max_chars:           Soft ceiling per chunk (default 500).  A single
                             sentence longer than this will be its own chunk.
        sentence_overlap:    Number of *sentences* (not chars) to carry over
                             into the next chunk for context continuity (default 1).
        split_on_paragraphs: If True, paragraph breaks (\\n\\n) always force a
                             new chunk regardless of the character limit (default True).
    """

    STRATEGY_NAME = "sentence"

    def __init__(
        self,
        max_chars: int = 500,
        sentence_overlap: int = 1,
        split_on_paragraphs: bool = True,
    ):
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if sentence_overlap < 0:
            raise ValueError("sentence_overlap must be >= 0")
        self.max_chars = max_chars
        self.sentence_overlap = sentence_overlap
        self.split_on_paragraphs = split_on_paragraphs

    # ------------------------------------------------------------------ #

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        parts = _SENTENCE_END_RE.split(text)
        return [part.strip() for part in parts if part.strip()]

    def chunk_document(self, document: Document) -> List[TextChunk]:
        text = document.content
        if not text.strip():
            return []

        if self.split_on_paragraphs:
            paragraphs = [p.strip() for p in _PARA_BREAK_RE.split(text) if p.strip()]
        else:
            paragraphs = [text.strip()]

        if not paragraphs:
            return []

        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_len = 0

        def flush(carry_over: int = 0) -> None:
            nonlocal current_sentences, current_len
            if not current_sentences:
                return
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(
                    _make_chunk(
                        text=chunk_text,
                        document_id=document.id,
                        index=len(chunks),
                        strategy=self.STRATEGY_NAME,
                        extra_meta={
                            "max_chars": self.max_chars,
                            "sentence_overlap": self.sentence_overlap,
                            "sentence_count": len(current_sentences),
                        },
                    )
                )
            # Keep the last `carry_over` sentences for overlap
            overlap_sents = current_sentences[-carry_over:] if carry_over else []
            current_sentences = list(overlap_sents)
            current_len = sum(len(s) for s in current_sentences)

        for para in paragraphs:
            sentences = self._split_sentences(para)
            for sent in sentences:
                sent_len = len(sent)
                if current_len + sent_len > self.max_chars and current_sentences:
                    flush(carry_over=self.sentence_overlap)
                current_sentences.append(sent)
                current_len += sent_len

            # Flush at paragraph boundary (hard boundary when split_on_paragraphs is True)
            flush(carry_over=0)

        return chunks


# ---------------------------------------------------------------------------
# Strategy 3 – Metadata-aware splitting
# ---------------------------------------------------------------------------

class MetadataAwareChunker(IChunker):
    """
    Adapts chunking behaviour based on the document's metadata.

    For MSMARCO-XI documents the relevant metadata fields are:
      - language     ("hi", "ta", "en", …) — determines sentence boundary chars
      - query_type   ("description" → paragraph-aware, "entity"/"numeric" → smaller)
      - is_selected  — marks the gold passage; recorded in chunk metadata

    The strategy records its reasoning in each TextChunk's metadata so the
    evaluation team can correlate chunk quality with source characteristics.

    Args:
        default_max_chars:   Fallback max chars if no metadata rule fires (500).
        description_max_chars: Max chars for "description" query types (700).
        entity_max_chars:    Max chars for "entity"/"numeric" query types (300).
    """

    STRATEGY_NAME = "metadata_aware"

    # ISO 639-1 codes that use Indic script and have their own sentence stops
    _INDIC_LANGS = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "ne",
                    "or", "pa", "sa", "ta", "te", "ur"}

    def __init__(
        self,
        default_max_chars: int = 500,
        description_max_chars: int = 700,
        entity_max_chars: int = 300,
    ):
        self.default_max_chars = default_max_chars
        self.description_max_chars = description_max_chars
        self.entity_max_chars = entity_max_chars

    # ------------------------------------------------------------------ #

    def _pick_max_chars(self, metadata: Dict[str, Any]) -> int:
        """Select the character limit based on query_type metadata."""
        qt = str(metadata.get("query_type", "")).lower()
        if qt == "description":
            return self.description_max_chars
        if qt in ("entity", "numeric", "person", "location"):
            return self.entity_max_chars
        return self.default_max_chars

    def _pick_sentence_overlap(self, metadata: Dict[str, Any]) -> int:
        """Use 2-sentence overlap for descriptive content, 1 otherwise."""
        qt = str(metadata.get("query_type", "")).lower()
        return 2 if qt == "description" else 1

    def chunk_document(self, document: Document) -> List[TextChunk]:
        text = document.content
        if not text.strip():
            return []

        meta = document.metadata
        language = str(meta.get("language", "")).lower()
        max_chars = self._pick_max_chars(meta)
        sentence_overlap = self._pick_sentence_overlap(meta)

        # For Indic languages, always enable paragraph splitting (text uses
        # ।/॥ as sentence markers, and paragraphs are meaningful units)
        split_on_paragraphs = language in self._INDIC_LANGS

        # Delegate to a SentenceChunker configured for this document
        inner = SentenceChunker(
            max_chars=max_chars,
            sentence_overlap=sentence_overlap,
            split_on_paragraphs=split_on_paragraphs,
        )
        base_chunks = inner.chunk_document(document)

        # Re-stamp strategy name and carry over source document metadata
        is_selected = meta.get("is_selected")
        result: List[TextChunk] = []
        for chunk in base_chunks:
            chunk.metadata["chunk_strategy"] = self.STRATEGY_NAME
            chunk.metadata["language"] = language or "unknown"
            chunk.metadata["query_type"] = meta.get("query_type", "")
            if is_selected is not None:
                chunk.metadata["is_selected"] = is_selected
            result.append(chunk)

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {
    FixedOverlapChunker.STRATEGY_NAME: FixedOverlapChunker,
    SentenceChunker.STRATEGY_NAME: SentenceChunker,
    MetadataAwareChunker.STRATEGY_NAME: MetadataAwareChunker,
}

STRATEGY_NAMES = list(_REGISTRY.keys())


def get_chunker(name: str, **kwargs: Any) -> IChunker:
    """
    Return a configured IChunker instance by strategy name.

    Args:
        name:   One of "fixed_overlap", "sentence", "metadata_aware".
        **kwargs: Forwarded to the chunker constructor.

    Raises:
        ValueError: If *name* is not a registered strategy.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown chunking strategy '{name}'. "
            f"Available: {STRATEGY_NAMES}"
        )
    return cls(**kwargs)
