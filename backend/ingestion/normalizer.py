"""
backend/ingestion/normalizer.py
--------------------------------
Stateless normalization functions for the AI4Bharat MSMARCO-XI dataset.

Raw schema (from ms_marco_translations.py):
    query_id            int   – original MS-MARCO query id
    query_type          str   – e.g. "description", "entity", …
    query               str   – TRANSLATED query
    Eng_Query           str   – original English query
    Answer              str   – TRANSLATED answer (may be "No Answer Present.")
    Eng_Answer          str   – original English answer
    source_lang         str   – e.g. "eng_Latn"
    target_lang         str   – e.g. "hin_Deva"
    meta                dict  – translation model metadata
    passages            dict  with keys:
        is_selected         list[int]  – 1 if the passage is the answer source
        English_passages    list[str]  – original English passage texts
        Translated_passages list[str]  – translated passage texts

One Document is produced per translated passage (RAG retrieval unit).
Stable document_id  =  "<query_id>_<lang>_p<passage_idx>"   (URL-safe, collision-free)
"""

import logging
from typing import Any, Dict, List, Optional

from backend.common.schemas import Document

logger = logging.getLogger(__name__)

# Languages supported by the dataset
SUPPORTED_LANGUAGES: List[str] = [
    "as", "bn", "gu", "hi", "kn", "ml", "mr",
    "ne", "or", "pa", "sa", "ta", "te", "ur",
]


def _safe_str(value: Any, default: str = "") -> str:
    """Coerce a value to str, returning *default* for None/NaN/empty."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int; return *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_record(
    raw: Dict[str, Any],
    language_code: str,
    split: str,
) -> List[Document]:
    """
    Normalize one raw HF row into a list of Documents (one per passage).

    Args:
        raw:           Dict with the raw field values from HF.
        language_code: ISO 639-1 code, e.g. "hi".
        split:         Dataset split, e.g. "train" or "validation".

    Returns:
        List of normalized Documents. Empty list if the record is malformed.
    """
    # --- Validate mandatory top-level fields ---
    query_id_raw = raw.get("query_id")
    if query_id_raw is None:
        logger.warning("Skipping record with missing query_id")
        return []

    query_id = _safe_int(query_id_raw)

    passages_block = raw.get("passages")
    if not passages_block:
        logger.warning("Skipping query_id=%s — missing passages block", query_id)
        return []

    translated_passages: List[str] = passages_block.get("Translated_passages") or []
    english_passages: List[str] = passages_block.get("English_passages") or []
    is_selected: List[int] = passages_block.get("is_selected") or []

    if not translated_passages:
        logger.warning("Skipping query_id=%s — empty Translated_passages", query_id)
        return []

    # --- Extract shared metadata ---
    query_text = _safe_str(raw.get("query"))
    eng_query = _safe_str(raw.get("Eng_Query"))
    answer = _safe_str(raw.get("Answer"))
    eng_answer = _safe_str(raw.get("Eng_Answer"))
    query_type = _safe_str(raw.get("query_type"))
    source_lang = _safe_str(raw.get("source_lang"))
    target_lang = _safe_str(raw.get("target_lang"))

    meta_block: Dict[str, Any] = raw.get("meta") or {}

    documents: List[Document] = []

    for idx, passage_text in enumerate(translated_passages):
        passage_text = _safe_str(passage_text)
        if not passage_text:
            logger.debug(
                "Skipping empty translated passage query_id=%s idx=%s", query_id, idx
            )
            continue

        doc_id = f"{query_id}_{language_code}_p{idx}"

        selected: Optional[int] = is_selected[idx] if idx < len(is_selected) else None
        eng_passage: str = (
            _safe_str(english_passages[idx]) if idx < len(english_passages) else ""
        )

        metadata: Dict[str, Any] = {
            # Source provenance
            "dataset": "ai4bharat/MSMARCO-XI",
            "language": language_code,
            "split": split,
            "source_lang": source_lang,
            "target_lang": target_lang,
            # Query context
            "query_id": query_id,
            "query_type": query_type,
            "query": query_text,
            "eng_query": eng_query,
            # Passage context
            "passage_index": idx,
            "is_selected": selected,
            "english_passage": eng_passage,
            # Answer context (may be "No Answer Present.")
            "answer": answer,
            "eng_answer": eng_answer,
            # Translation model provenance
            "translation_model": _safe_str(meta_block.get("model_name")),
        }

        documents.append(
            Document(id=doc_id, content=passage_text, metadata=metadata)
        )

    return documents
