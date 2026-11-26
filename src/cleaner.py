"""Regex-based OCR noise cleaner preserving technical terminology."""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_CONTROL_AND_BROKEN_UNICODE_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\u200b-\u200d\ufeff]"
)
_DECORATIVE_LINE_RE = re.compile(r"[_\-=~*]{3,}")
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
_ALL_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean OCR artifacts and normalize whitespace while preserving technical terms.

    Removes control characters, zero-width/replacement Unicode artifacts, decorative
    divider lines (e.g., ``____``, ``====``, ``----``), and redundant whitespace.
    Explicitly preserves characters used in technical identifiers such as ``+``, ``#``,
    ``.``, ``/``, single hyphens, digits, and Turkish Unicode letters.

    Args:
        text: Raw extracted text from PDF, DOCX, or OCR.

    Returns:
        Cleaned text string, or an empty string if no content remains.
    """
    if not text or not text.strip():
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_AND_BROKEN_UNICODE_RE.sub("", normalized)
    normalized = _DECORATIVE_LINE_RE.sub("", normalized)

    cleaned_lines = [
        _HORIZONTAL_SPACE_RE.sub(" ", line).strip()
        for line in normalized.split("\n")
    ]
    joined = "\n".join(cleaned_lines)
    joined = _EXCESS_NEWLINES_RE.sub("\n\n", joined).strip()

    logger.debug("Cleaned text from %d to %d characters.", len(text), len(joined))
    return joined


def normalize_for_comparison(text: str) -> str:
    """Normalize text identically for fair WER/CER benchmark comparison.

    Applies :func:`clean_text`, lowercases all characters, and collapses all
    whitespace runs (including newlines) into single spaces.

    Args:
        text: Ground truth or OCR hypothesis text.

    Returns:
        Normalized single-line lowercase text string.
    """
    cleaned = clean_text(text).lower()
    return _ALL_WHITESPACE_RE.sub(" ", cleaned).strip()
