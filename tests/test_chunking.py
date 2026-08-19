"""
tests/test_chunking.py
------------------------
Unit tests for all three chunking strategies.

Tests verify:
  1. Schema conformance  – every TextChunk has text, document_id, metadata.
  2. Mandatory metadata  – chunk_strategy and chunk_index always present.
  3. No empty chunks     – text field is always non-empty after stripping.
  4. No duplicate chunks – (document_id, chunk_index) pairs are unique.
  5. Coverage check      – every character of the original text appears in
                           at least one chunk (character-set coverage).
  6. Strategy tag        – metadata.chunk_strategy matches the implementation.
  7. Document ID pass-through – chunk.document_id == parent document.id.
  8. Edge cases          – empty document, single-sentence document, very
                           long single word (no natural split point).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from backend.chunking.strategies import (
    FixedOverlapChunker,
    MetadataAwareChunker,
    SentenceChunker,
    get_chunker,
    STRATEGY_NAMES,
)
from backend.common.schemas import Document, TextChunk

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SHORT_EN_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump!"
)

LONG_EN_TEXT = (
    "Retrieval-Augmented Generation (RAG) is a technique that combines "
    "information retrieval with language generation. "
    "It was introduced to address the knowledge limitations of large language models (LLMs). "
    "In a RAG system, a retriever first fetches relevant documents from a corpus. "
    "Then a generator (LLM) uses those documents as context to produce a factual answer. "
    "This approach reduces hallucinations because the model is grounded in retrieved evidence. "
    "RAG has become popular in enterprise knowledge management and customer support systems. "
    "Variants include dense passage retrieval (DPR), BM25-based sparse retrieval, and hybrid methods. "
    "The quality of retrieved passages critically determines the quality of the final answer. "
    "Chunking strategies therefore play a central role in RAG pipeline performance.\n\n"
    "Sentence boundaries are important in multilingual settings. "
    "Hindi text uses the danda (।) as a sentence terminator. "
    "Properly respecting these boundaries preserves semantic coherence within each chunk. "
    "A fixed character split may cut a sentence mid-way, degrading retrieval precision."
)

HINDI_TEXT = (
    "RAG एक तकनीक है जो सूचना पुनर्प्राप्ति को भाषा पीढ़ी के साथ जोड़ती है। "
    "इसे बड़े भाषा मॉडल की ज्ञान सीमाओं को दूर करने के लिए पेश किया गया था। "
    "RAG सिस्टम में, एक रिट्रीवर पहले कॉर्पस से प्रासंगिक दस्तावेज़ लाता है। "
    "फिर एक जनरेटर उन दस्तावेज़ों को संदर्भ के रूप में उपयोग करके एक तथ्यात्मक उत्तर देता है।"
)

MULTI_PARA_TEXT = (
    "First paragraph line one. First paragraph line two. "
    "First paragraph line three.\n\n"
    "Second paragraph line one. Second paragraph line two.\n\n"
    "Third paragraph, single line."
)


def make_doc(
    content: str,
    doc_id: str = "test_doc_1",
    metadata: dict | None = None,
) -> Document:
    return Document(id=doc_id, content=content, metadata=metadata or {})


# ---------------------------------------------------------------------------
# Generic contract checks (applied to every strategy)
# ---------------------------------------------------------------------------

def assert_contract(chunks: List[TextChunk], doc: Document, strategy_name: str) -> None:
    """Run the full cross-team schema contract assertions."""
    assert isinstance(chunks, list), "chunk_document must return a list"

    for chunk in chunks:
        # ── Schema conformance ──────────────────────────────────────────
        assert isinstance(chunk, TextChunk), f"Each item must be a TextChunk, got {type(chunk)}"
        assert hasattr(chunk, "text") and isinstance(chunk.text, str)
        assert hasattr(chunk, "document_id") and isinstance(chunk.document_id, str)
        assert hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict)

        # ── No empty chunks ─────────────────────────────────────────────
        assert chunk.text.strip(), f"Empty text in chunk: {chunk!r}"

        # ── Document ID pass-through ─────────────────────────────────────
        assert chunk.document_id == doc.id, (
            f"document_id mismatch: expected {doc.id!r}, got {chunk.document_id!r}"
        )

        # ── Mandatory metadata keys ──────────────────────────────────────
        assert "chunk_strategy" in chunk.metadata, "metadata.chunk_strategy missing"
        assert "chunk_index" in chunk.metadata, "metadata.chunk_index missing"
        assert chunk.metadata["chunk_strategy"] == strategy_name, (
            f"chunk_strategy tag wrong: expected {strategy_name!r}, "
            f"got {chunk.metadata['chunk_strategy']!r}"
        )

    # ── No duplicate (doc_id, chunk_index) pairs ────────────────────────
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert len(indices) == len(set(indices)), "Duplicate chunk_index values found"

    # ── chunk_index is contiguous from 0 ────────────────────────────────
    if chunks:
        assert sorted(indices) == list(range(len(chunks))), (
            f"chunk_index values are not contiguous from 0: {sorted(indices)}"
        )


def assert_coverage(chunks: List[TextChunk], doc: Document) -> None:
    """
    Verify that every character in the original text is covered by at least
    one chunk (character-set coverage, not exact positional coverage).

    We use a set-of-characters check rather than positional because overlapping
    chunkers and sentence chunkers may reorder whitespace.  The key invariant
    is: no content is silently dropped.
    """
    if not doc.content.strip():
        return  # empty doc → nothing to cover

    original_chars = set(doc.content)
    covered_chars = set()
    for chunk in chunks:
        covered_chars.update(chunk.text)

    # Every non-whitespace character in the original must appear somewhere
    uncovered = original_chars - covered_chars - {" ", "\n", "\t", "\r"}
    assert not uncovered, (
        f"Characters present in source doc but absent from all chunks: "
        f"{sorted(uncovered)!r}"
    )


# ---------------------------------------------------------------------------
# FixedOverlapChunker
# ---------------------------------------------------------------------------

class TestFixedOverlapChunker:

    def _chunker(self, chunk_size=100, overlap=20):
        return FixedOverlapChunker(chunk_size=chunk_size, overlap=overlap)

    def test_strategy_name_tag(self):
        doc = make_doc(LONG_EN_TEXT)
        chunks = self._chunker().chunk_document(doc)
        for c in chunks:
            assert c.metadata["chunk_strategy"] == "fixed_overlap"

    def test_full_contract_short_text(self):
        doc = make_doc(SHORT_EN_TEXT)
        chunks = self._chunker(chunk_size=80, overlap=10).chunk_document(doc)
        assert_contract(chunks, doc, "fixed_overlap")

    def test_full_contract_long_text(self):
        doc = make_doc(LONG_EN_TEXT)
        chunks = self._chunker(chunk_size=200, overlap=30).chunk_document(doc)
        assert_contract(chunks, doc, "fixed_overlap")

    def test_coverage_long_text(self):
        doc = make_doc(LONG_EN_TEXT)
        chunks = self._chunker(chunk_size=200, overlap=30).chunk_document(doc)
        assert_coverage(chunks, doc)

    def test_overlap_metadata_recorded(self):
        doc = make_doc(LONG_EN_TEXT)
        chunks = self._chunker(chunk_size=150, overlap=25).chunk_document(doc)
        assert all(c.metadata["overlap"] == 25 for c in chunks)
        assert all(c.metadata["chunk_size"] == 150 for c in chunks)

    def test_char_positions_non_overlapping_in_simple_case(self):
        """With overlap=0 the windows should be strictly adjacent."""
        doc = make_doc("A" * 300)
        chunks = FixedOverlapChunker(chunk_size=100, overlap=0).chunk_document(doc)
        assert len(chunks) == 3
        for c in chunks:
            assert len(c.text) == 100

    def test_single_chunk_when_text_fits(self):
        doc = make_doc("Hello world.")
        chunks = self._chunker(chunk_size=500, overlap=50).chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_empty_document_returns_empty_list(self):
        doc = make_doc("")
        chunks = self._chunker().chunk_document(doc)
        assert chunks == []

    def test_whitespace_only_document_returns_empty_list(self):
        doc = make_doc("   \n\n  ")
        chunks = self._chunker().chunk_document(doc)
        assert chunks == []

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            FixedOverlapChunker(chunk_size=0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedOverlapChunker(chunk_size=100, overlap=100)  # overlap >= chunk_size

    def test_document_id_matches(self):
        doc = make_doc(LONG_EN_TEXT, doc_id="my_special_id")
        chunks = self._chunker().chunk_document(doc)
        assert all(c.document_id == "my_special_id" for c in chunks)


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------

class TestSentenceChunker:

    def _chunker(self, max_chars=300, sentence_overlap=1, split_on_paragraphs=True):
        return SentenceChunker(
            max_chars=max_chars,
            sentence_overlap=sentence_overlap,
            split_on_paragraphs=split_on_paragraphs,
        )

    def test_strategy_name_tag(self):
        doc = make_doc(LONG_EN_TEXT)
        chunks = self._chunker().chunk_document(doc)
        for c in chunks:
            assert c.metadata["chunk_strategy"] == "sentence"

    def test_full_contract_english(self):
        doc = make_doc(LONG_EN_TEXT)
        chunks = self._chunker(max_chars=300).chunk_document(doc)
        assert_contract(chunks, doc, "sentence")

    def test_full_contract_hindi(self):
        doc = make_doc(HINDI_TEXT)
        chunks = self._chunker(max_chars=200).chunk_document(doc)
        assert_contract(chunks, doc, "sentence")

    def test_coverage_english(self):
        doc = make_doc(LONG_EN_TEXT)
        chunks = self._chunker(max_chars=300).chunk_document(doc)
        assert_coverage(chunks, doc)

    def test_coverage_hindi(self):
        doc = make_doc(HINDI_TEXT)
        chunks = self._chunker(max_chars=200).chunk_document(doc)
        assert_coverage(chunks, doc)

    def test_paragraph_break_forces_new_chunk(self):
        doc = make_doc(MULTI_PARA_TEXT)
        chunks = self._chunker(max_chars=1000, split_on_paragraphs=True).chunk_document(doc)
        # With a very large max_chars the only split trigger is the para break.
        # There are 3 paragraphs so we expect >= 3 chunks (overlap may merge some).
        assert len(chunks) >= 3

    def test_no_paragraph_split_consolidates(self):
        doc = make_doc(MULTI_PARA_TEXT)
        chunks_no_para = self._chunker(max_chars=1000, split_on_paragraphs=False).chunk_document(doc)
        chunks_para = self._chunker(max_chars=1000, split_on_paragraphs=True).chunk_document(doc)
        # Without paragraph splitting fewer chunks should result
        assert len(chunks_no_para) <= len(chunks_para)

    def test_single_long_sentence_becomes_one_chunk(self):
        """A sentence longer than max_chars must not be silently dropped."""
        long_sentence = "word " * 200  # ~1000 chars
        doc = make_doc(long_sentence.strip() + ".")
        chunks = self._chunker(max_chars=100).chunk_document(doc)
        assert len(chunks) >= 1
        assert_coverage(chunks, doc)

    def test_empty_document(self):
        doc = make_doc("")
        assert self._chunker().chunk_document(doc) == []

    def test_single_sentence_document(self):
        doc = make_doc("Just one sentence.")
        chunks = self._chunker(max_chars=500).chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == "Just one sentence."

    def test_sentence_count_metadata_present(self):
        doc = make_doc(LONG_EN_TEXT)
        chunks = self._chunker().chunk_document(doc)
        for c in chunks:
            assert "sentence_count" in c.metadata
            assert c.metadata["sentence_count"] >= 1

    def test_invalid_max_chars_raises(self):
        with pytest.raises(ValueError):
            SentenceChunker(max_chars=0)


# ---------------------------------------------------------------------------
# MetadataAwareChunker
# ---------------------------------------------------------------------------

class TestMetadataAwareChunker:

    def _make_hi_doc(self, content=None, query_type="description"):
        return make_doc(
            HINDI_TEXT if content is None else content,
            metadata={"language": "hi", "query_type": query_type},
        )

    def _make_en_doc(self, content=None, query_type="entity"):
        return make_doc(
            SHORT_EN_TEXT if content is None else content,
            metadata={"language": "en", "query_type": query_type},
        )

    def test_strategy_name_tag(self):
        doc = self._make_hi_doc()
        chunks = MetadataAwareChunker().chunk_document(doc)
        for c in chunks:
            assert c.metadata["chunk_strategy"] == "metadata_aware"

    def test_full_contract_hindi_description(self):
        doc = self._make_hi_doc(content=LONG_EN_TEXT, query_type="description")
        chunks = MetadataAwareChunker().chunk_document(doc)
        assert_contract(chunks, doc, "metadata_aware")

    def test_full_contract_entity(self):
        doc = self._make_en_doc(query_type="entity")
        chunks = MetadataAwareChunker().chunk_document(doc)
        assert_contract(chunks, doc, "metadata_aware")

    def test_coverage_hindi(self):
        doc = self._make_hi_doc(content=HINDI_TEXT)
        chunks = MetadataAwareChunker().chunk_document(doc)
        assert_coverage(chunks, doc)

    def test_description_produces_larger_chunks_than_entity(self):
        """description max_chars > entity max_chars → fewer chunks from same text."""
        content = LONG_EN_TEXT
        doc_desc = make_doc(content, metadata={"query_type": "description", "language": "en"})
        doc_ent = make_doc(content, metadata={"query_type": "entity", "language": "en"})
        chunker = MetadataAwareChunker(description_max_chars=700, entity_max_chars=200)
        chunks_desc = chunker.chunk_document(doc_desc)
        chunks_ent = chunker.chunk_document(doc_ent)
        assert len(chunks_desc) <= len(chunks_ent), (
            "Description chunking should produce fewer (larger) chunks than entity"
        )

    def test_language_and_query_type_propagated_to_metadata(self):
        doc = self._make_hi_doc(query_type="numeric")
        chunks = MetadataAwareChunker().chunk_document(doc)
        for c in chunks:
            assert c.metadata["language"] == "hi"
            assert c.metadata["query_type"] == "numeric"

    def test_is_selected_propagated_when_present(self):
        doc = make_doc(
            HINDI_TEXT,
            metadata={"language": "hi", "query_type": "description", "is_selected": 1},
        )
        chunks = MetadataAwareChunker().chunk_document(doc)
        for c in chunks:
            assert c.metadata["is_selected"] == 1

    def test_empty_document(self):
        doc = self._make_hi_doc(content="")
        assert MetadataAwareChunker().chunk_document(doc) == []

    def test_no_metadata_falls_back_to_defaults(self):
        doc = make_doc(LONG_EN_TEXT, metadata={})
        chunks = MetadataAwareChunker().chunk_document(doc)
        assert_contract(chunks, doc, "metadata_aware")
        assert_coverage(chunks, doc)


# ---------------------------------------------------------------------------
# Factory + get_chunker
# ---------------------------------------------------------------------------

class TestFactory:

    def test_all_registered_strategies_instantiate(self):
        for name in STRATEGY_NAMES:
            chunker = get_chunker(name)
            assert isinstance(chunker, object)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent_strategy")

    def test_kwargs_forwarded(self):
        chunker = get_chunker("fixed_overlap", chunk_size=200, overlap=10)
        assert chunker.chunk_size == 200
        assert chunker.overlap == 10

    def test_all_strategies_produce_valid_chunks_on_long_text(self):
        doc = make_doc(LONG_EN_TEXT, metadata={"language": "en", "query_type": "description"})
        for name in STRATEGY_NAMES:
            chunker = get_chunker(name)
            chunks = chunker.chunk_document(doc)
            assert_contract(chunks, doc, name)
            assert_coverage(chunks, doc)


# ---------------------------------------------------------------------------
# chunk.py script integration test
# ---------------------------------------------------------------------------

class TestChunkScript:
    """End-to-end test of chunk.py CLI using temporary JSONL files."""

    def _write_doc_jsonl(self, path: Path, docs) -> None:
        with path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc.model_dump(), ensure_ascii=False) + "\n")

    def test_produces_valid_textchunk_jsonl(self, tmp_path):
        from backend.scripts.chunk import run, parse_args

        input_file = tmp_path / "docs.jsonl"
        output_file = tmp_path / "chunks.jsonl"
        self._write_doc_jsonl(input_file, [
            make_doc(LONG_EN_TEXT, "doc1"),
            make_doc(SHORT_EN_TEXT, "doc2"),
        ])

        args = parse_args([
            "--input", str(input_file),
            "--output", str(output_file),
            "--strategy", "sentence",
            "--max-chars", "300",
        ])
        exit_code = run(args)
        assert exit_code == 0
        assert output_file.exists()

        lines = [l for l in output_file.read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj and obj["text"].strip()
            assert "document_id" in obj
            assert "metadata" in obj
            assert obj["metadata"]["chunk_strategy"] == "sentence"

    def test_idempotent_rerun(self, tmp_path):
        from backend.scripts.chunk import run, parse_args

        input_file = tmp_path / "docs.jsonl"
        output_file = tmp_path / "chunks.jsonl"
        self._write_doc_jsonl(input_file, [make_doc(LONG_EN_TEXT, "doc1")])

        common = ["--input", str(input_file), "--output", str(output_file),
                  "--strategy", "fixed_overlap"]
        run(parse_args(common))
        first = output_file.read_text("utf-8")
        run(parse_args(common))
        second = output_file.read_text("utf-8")
        assert first == second

    def test_all_strategies_via_cli(self, tmp_path):
        from backend.scripts.chunk import run, parse_args

        input_file = tmp_path / "docs.jsonl"
        self._write_doc_jsonl(input_file, [make_doc(LONG_EN_TEXT, "doc1")])

        for strategy in STRATEGY_NAMES:
            output_file = tmp_path / f"chunks_{strategy}.jsonl"
            args = parse_args([
                "--input", str(input_file),
                "--output", str(output_file),
                "--strategy", strategy,
            ])
            exit_code = run(args)
            assert exit_code == 0, f"chunk.py failed for strategy={strategy}"
            lines = [l for l in output_file.read_text("utf-8").splitlines() if l.strip()]
            assert len(lines) > 0, f"No output for strategy={strategy}"
