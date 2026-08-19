"""
tests/test_ingestion.py
------------------------
Unit tests for the ingestion layer.

All tests run against the local fixture file (tests/fixtures/msmarco_sample.jsonl)
and never touch the network or the full HF dataset.

Fixture summary (5 records):
  query_id 1001 → 2 passages (normal)
  query_id 1002 → 1 passage  (normal)
  query_id 1003 → 1 passage  (null query field — should still normalise)
  query_id 1004 → 0 passages (empty list — should be skipped)
  query_id null  → skipped by normalizer (missing query_id)

Expected output: 4 Documents total.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from backend.common.schemas import Document
from backend.ingestion.loader import MSMarcoXILoader
from backend.ingestion.normalizer import normalize_record, SUPPORTED_LANGUAGES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "msmarco_sample.jsonl"
LANGUAGE = "hi"
SPLIT = "train"


def load_fixture_docs() -> list[Document]:
    """Load all Documents from the local JSONL fixture."""
    loader = MSMarcoXILoader(language=LANGUAGE, split=SPLIT)
    return loader.load(str(FIXTURE_PATH))


# ---------------------------------------------------------------------------
# normalizer unit tests
# ---------------------------------------------------------------------------

class TestNormalizeRecord:
    """Unit tests for the stateless normalize_record() function."""

    def _raw_record(self, **overrides):
        base = {
            "query_id": 9001,
            "query_type": "description",
            "query": "परीक्षण प्रश्न",
            "Eng_Query": "Test question",
            "Answer": "परीक्षण उत्तर",
            "Eng_Answer": "Test answer",
            "source_lang": "eng_Latn",
            "target_lang": "hin_Deva",
            "meta": {
                "model_name": "gpt-4o",
                "temperature": 0.0,
                "max_tokens": 1024,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            "passages": {
                "is_selected": [1, 0],
                "English_passages": ["English passage 1.", "English passage 2."],
                "Translated_passages": ["हिंदी अनुच्छेद 1।", "हिंदी अनुच्छेद 2।"],
            },
        }
        base.update(overrides)
        return base

    def test_normal_record_produces_correct_document_count(self):
        docs = normalize_record(self._raw_record(), LANGUAGE, SPLIT)
        assert len(docs) == 2, "Expected one Document per translated passage"

    def test_document_id_is_stable_and_deterministic(self):
        docs = normalize_record(self._raw_record(), LANGUAGE, SPLIT)
        assert docs[0].id == "9001_hi_p0"
        assert docs[1].id == "9001_hi_p1"

    def test_document_content_is_translated_passage(self):
        docs = normalize_record(self._raw_record(), LANGUAGE, SPLIT)
        assert docs[0].content == "हिंदी अनुच्छेद 1।"
        assert docs[1].content == "हिंदी अनुच्छेद 2।"

    def test_metadata_fields_present(self):
        docs = normalize_record(self._raw_record(), LANGUAGE, SPLIT)
        meta = docs[0].metadata
        required_keys = {
            "dataset", "language", "split", "source_lang", "target_lang",
            "query_id", "query_type", "query", "eng_query",
            "passage_index", "is_selected", "english_passage",
            "answer", "eng_answer", "translation_model",
        }
        assert required_keys.issubset(meta.keys()), (
            f"Missing metadata keys: {required_keys - meta.keys()}"
        )

    def test_metadata_values_correct(self):
        docs = normalize_record(self._raw_record(), LANGUAGE, SPLIT)
        meta = docs[0].metadata
        assert meta["query_id"] == 9001
        assert meta["language"] == LANGUAGE
        assert meta["split"] == SPLIT
        assert meta["is_selected"] == 1
        assert meta["passage_index"] == 0
        assert meta["dataset"] == "ai4bharat/MSMARCO-XI"
        assert meta["translation_model"] == "gpt-4o"

    def test_second_passage_is_selected_false(self):
        docs = normalize_record(self._raw_record(), LANGUAGE, SPLIT)
        assert docs[1].metadata["is_selected"] == 0

    def test_missing_query_id_skips_record(self):
        raw = self._raw_record()
        raw["query_id"] = None
        docs = normalize_record(raw, LANGUAGE, SPLIT)
        assert docs == [], "Record with null query_id should be skipped"

    def test_empty_translated_passages_skips_record(self):
        raw = self._raw_record()
        raw["passages"]["Translated_passages"] = []
        docs = normalize_record(raw, LANGUAGE, SPLIT)
        assert docs == [], "Record with empty passages should produce no Documents"

    def test_missing_passages_block_skips_record(self):
        raw = self._raw_record()
        raw["passages"] = None
        docs = normalize_record(raw, LANGUAGE, SPLIT)
        assert docs == []

    def test_null_query_field_normalised_to_empty_string(self):
        raw = self._raw_record()
        raw["query"] = None
        docs = normalize_record(raw, LANGUAGE, SPLIT)
        assert len(docs) == 2
        assert docs[0].metadata["query"] == ""

    def test_empty_passage_string_skipped(self):
        raw = self._raw_record()
        raw["passages"]["Translated_passages"] = ["", "हिंदी अनुच्छेद 2।"]
        docs = normalize_record(raw, LANGUAGE, SPLIT)
        assert len(docs) == 1
        assert docs[0].id == "9001_hi_p1"

    def test_documents_are_document_schema_instances(self):
        docs = normalize_record(self._raw_record(), LANGUAGE, SPLIT)
        for doc in docs:
            assert isinstance(doc, Document)


# ---------------------------------------------------------------------------
# loader integration tests (using local fixture file)
# ---------------------------------------------------------------------------

class TestMSMarcoXILoader:
    """Integration tests for MSMarcoXILoader reading from the local fixture."""

    def test_load_returns_list_of_documents(self):
        docs = load_fixture_docs()
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_correct_document_count(self):
        """
        Fixture: 5 records
          1001 → 2 passages ✓
          1002 → 1 passage  ✓
          1003 → 1 passage  ✓ (null query field normalised gracefully)
          1004 → 0 passages ✗ (empty list)
          null → skipped    ✗ (missing query_id)
        Expected: 4 Documents
        """
        docs = load_fixture_docs()
        assert len(docs) == 4

    def test_all_ids_are_unique(self):
        docs = load_fixture_docs()
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids)), "Duplicate document IDs detected"

    def test_id_format_matches_pattern(self):
        """IDs must match <query_id>_<lang>_p<idx>"""
        import re
        pattern = re.compile(r"^\d+_[a-z]{2}_p\d+$")
        docs = load_fixture_docs()
        for doc in docs:
            assert pattern.match(doc.id), f"Unexpected id format: {doc.id!r}"

    def test_content_is_non_empty_string(self):
        docs = load_fixture_docs()
        for doc in docs:
            assert isinstance(doc.content, str)
            assert len(doc.content) > 0

    def test_required_metadata_keys_present_on_all_docs(self):
        required = {
            "dataset", "language", "split", "query_id",
            "passage_index", "is_selected",
        }
        docs = load_fixture_docs()
        for doc in docs:
            missing = required - doc.metadata.keys()
            assert not missing, f"Doc {doc.id} missing metadata keys: {missing}"

    def test_language_metadata_matches_loader_language(self):
        docs = load_fixture_docs()
        for doc in docs:
            assert doc.metadata["language"] == LANGUAGE

    def test_split_metadata_matches_loader_split(self):
        docs = load_fixture_docs()
        for doc in docs:
            assert doc.metadata["split"] == SPLIT

    def test_limit_respected(self):
        """limit=1 raw row → only passages from the first record are returned."""
        loader = MSMarcoXILoader(language=LANGUAGE, split=SPLIT, limit=1)
        docs = loader.load(str(FIXTURE_PATH))
        # First fixture record (query_id 1001) has 2 passages
        assert len(docs) == 2
        assert all(d.metadata["query_id"] == 1001 for d in docs)

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="not in supported set"):
            MSMarcoXILoader(language="xx")


# ---------------------------------------------------------------------------
# ingest.py script integration test (runs end-to-end via run())
# ---------------------------------------------------------------------------

class TestIngestScript:
    """End-to-end test of the ingest CLI using the local fixture."""

    def test_produces_valid_jsonl_output(self, tmp_path):
        from backend.scripts.ingest import run, parse_args

        output_file = tmp_path / "out.jsonl"
        args = parse_args([
            "--language", LANGUAGE,
            "--split", SPLIT,
            "--source-path", str(FIXTURE_PATH),
            "--output", str(output_file),
        ])

        exit_code = run(args)
        assert exit_code == 0
        assert output_file.exists()

        lines = [l for l in output_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 4  # same as load_fixture_docs()

        for line in lines:
            obj = json.loads(line)
            assert "id" in obj
            assert "content" in obj
            assert "metadata" in obj

    def test_idempotent_rerun_overwrites_cleanly(self, tmp_path):
        """Running the script twice should produce identical output."""
        from backend.scripts.ingest import run, parse_args

        output_file = tmp_path / "out.jsonl"
        common_args = [
            "--language", LANGUAGE,
            "--split", SPLIT,
            "--source-path", str(FIXTURE_PATH),
            "--output", str(output_file),
        ]

        run(parse_args(common_args))
        first_content = output_file.read_text(encoding="utf-8")

        run(parse_args(common_args))
        second_content = output_file.read_text(encoding="utf-8")

        assert first_content == second_content, "Re-run produced different output (not idempotent)"

    def test_output_dir_created_if_missing(self, tmp_path):
        from backend.scripts.ingest import run, parse_args

        nested = tmp_path / "deep" / "nested"
        output_file = nested / "out.jsonl"

        args = parse_args([
            "--language", LANGUAGE,
            "--split", SPLIT,
            "--source-path", str(FIXTURE_PATH),
            "--output", str(output_file),
        ])
        exit_code = run(args)
        assert exit_code == 0
        assert output_file.exists()


# ---------------------------------------------------------------------------
# DuckDB path tests (mocked — no network)
# ---------------------------------------------------------------------------

class TestDuckDBLoader:
    """
    Tests for the DuckDB-based remote loading path (_iter_hf_streaming).

    The DuckDB connection is monkey-patched so no network I/O occurs.
    We verify that:
      1. DuckDB SQL is called with the correct URL and LIMIT clause.
      2. Rows yielded by the new path have the exact shape expected by
         normalize_record() (same keys, same nested structure).
      3. normalize_record() successfully converts those rows into Documents
         without modification.
    """

    # A single representative raw row matching the MSMARCO-XI parquet schema.
    # DuckDB returns nested structs as plain Python dicts, which is what we
    # reproduce here.
    SAMPLE_ROW = {
        "query_id": 5001,
        "query_type": "description",
        "query": "डकDB क्या है?",
        "Eng_Query": "What is DuckDB?",
        "Answer": "एक OLAP डेटाबेस।",
        "Eng_Answer": "An OLAP database.",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "meta": {
            "model_name": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 1024,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "passages": {
            "is_selected": [1, 0],
            "English_passages": ["DuckDB is an OLAP DB.", "Another passage."],
            "Translated_passages": ["डकDB एक OLAP डेटाबेस है।", "एक और अनुच्छेद।"],
        },
    }

    def _make_mock_con(self, rows, col_names):
        """
        Build a mock DuckDB connection whose execute() returns a cursor-like
        object with .description and .fetchall() matching *rows*.
        """
        from unittest.mock import MagicMock

        # Mock cursor returned by con.execute(sql)
        mock_cursor = MagicMock()
        mock_cursor.description = [(c,) for c in col_names]
        mock_cursor.fetchall.return_value = [
            tuple(r[c] for c in col_names) for r in rows
        ]

        # con.execute() always returns mock_cursor (covers INSTALL/LOAD + SELECT)
        mock_con = MagicMock()
        mock_con.execute.return_value = mock_cursor

        return mock_con

    def test_duckdb_path_yields_correctly_shaped_rows(self, monkeypatch):
        """_iter_hf_streaming yields dicts with all expected top-level keys."""
        col_names = list(self.SAMPLE_ROW.keys())
        mock_con = self._make_mock_con([self.SAMPLE_ROW], col_names)

        import duckdb
        monkeypatch.setattr(duckdb, "connect", lambda: mock_con)

        loader = MSMarcoXILoader(language=LANGUAGE, split=SPLIT, limit=1)
        rows = list(loader._iter_hf_streaming())

        assert len(rows) == 1
        row = rows[0]
        for key in self.SAMPLE_ROW:
            assert key in row, f"Missing key '{key}' in yielded row"

    def test_duckdb_sql_includes_limit_clause(self, monkeypatch):
        """When self.limit > 0, the SQL query must contain a LIMIT clause."""
        col_names = list(self.SAMPLE_ROW.keys())
        mock_con = self._make_mock_con([self.SAMPLE_ROW], col_names)

        import duckdb
        monkeypatch.setattr(duckdb, "connect", lambda: mock_con)

        loader = MSMarcoXILoader(language=LANGUAGE, split=SPLIT, limit=10)
        list(loader._iter_hf_streaming())  # consume to trigger the execute call

        # Collect all SQL strings passed to con.execute()
        sql_calls = [call.args[0] for call in mock_con.execute.call_args_list]
        select_calls = [s for s in sql_calls if s.strip().upper().startswith("SELECT")]
        assert select_calls, "No SELECT statement found in DuckDB execute calls"
        assert "LIMIT 10" in select_calls[0], (
            f"Expected 'LIMIT 10' in SQL, got: {select_calls[0]!r}"
        )

    def test_duckdb_sql_omits_limit_clause_when_zero(self, monkeypatch):
        """When self.limit == 0 (no cap), the SQL must NOT contain LIMIT."""
        col_names = list(self.SAMPLE_ROW.keys())
        mock_con = self._make_mock_con([self.SAMPLE_ROW], col_names)

        import duckdb
        monkeypatch.setattr(duckdb, "connect", lambda: mock_con)

        loader = MSMarcoXILoader(language=LANGUAGE, split=SPLIT, limit=0)
        list(loader._iter_hf_streaming())

        sql_calls = [call.args[0] for call in mock_con.execute.call_args_list]
        select_calls = [s for s in sql_calls if s.strip().upper().startswith("SELECT")]
        assert select_calls
        assert "LIMIT" not in select_calls[0].upper(), (
            f"Unexpected LIMIT in uncapped query: {select_calls[0]!r}"
        )

    def test_duckdb_rows_flow_through_normalize_record(self, monkeypatch):
        """End-to-end: DuckDB row → normalize_record() → valid Documents."""
        col_names = list(self.SAMPLE_ROW.keys())
        mock_con = self._make_mock_con([self.SAMPLE_ROW], col_names)

        import duckdb
        monkeypatch.setattr(duckdb, "connect", lambda: mock_con)

        loader = MSMarcoXILoader(language=LANGUAGE, split=SPLIT, limit=1)
        docs = list(loader._iter_documents(LANGUAGE))  # triggers DuckDB path

        assert len(docs) == 2  # SAMPLE_ROW has 2 translated passages
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.metadata["query_id"] == 5001
            assert doc.metadata["language"] == LANGUAGE

    def test_duckdb_connection_closed_on_success(self, monkeypatch):
        """con.close() must be called even after successful iteration."""
        col_names = list(self.SAMPLE_ROW.keys())
        mock_con = self._make_mock_con([self.SAMPLE_ROW], col_names)

        import duckdb
        monkeypatch.setattr(duckdb, "connect", lambda: mock_con)

        loader = MSMarcoXILoader(language=LANGUAGE, split=SPLIT, limit=1)
        list(loader._iter_hf_streaming())

        mock_con.close.assert_called_once()
