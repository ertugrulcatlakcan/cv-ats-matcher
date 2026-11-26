"""Shared data contracts and exception types for Phase 1 & Phase 2 integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ExtractionMethod = Literal["direct_pdf", "ocr_pdf", "ocr_image", "docx"]


@dataclass
class ExtractionResult:
    """Structured output contract produced by the CV extraction pipeline.

    Attributes:
        file_path: Absolute or normalized string path to the source document.
        filename: Base filename of the processed document.
        method: Extraction route used (``direct_pdf``, ``ocr_pdf``, ``ocr_image``, or ``docx``).
        page_count: Number of pages processed (1 for single images and reflowable DOCX).
        raw_text: Uncleaned text directly extracted by PDF parser, DOCX reader, or OCR.
        cleaned_text: Noise-cleaned text preserving technical terminology.
        processing_time_sec: Elapsed wall-clock processing duration in seconds.
        warnings: Non-fatal diagnostic warnings encountered during extraction.
    """

    file_path: str
    filename: str
    method: ExtractionMethod
    page_count: int
    raw_text: str
    cleaned_text: str
    processing_time_sec: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the extraction result into a JSON-compatible dictionary.

        Returns:
            Dictionary representation matching all dataclass fields.
        """
        return asdict(self)


class ExtractionError(Exception):
    """Base exception for all CV extraction pipeline failures."""


class UnsupportedFileTypeError(ExtractionError):
    """Raised when an input file has an unsupported extension."""


class CorruptedFileError(ExtractionError):
    """Raised when a file cannot be parsed or decoded due to corruption."""


class EncryptedPDFError(ExtractionError):
    """Raised when a PDF is encrypted and requires a password."""


class TesseractNotFoundError(ExtractionError):
    """Raised when Tesseract OCR executable cannot be located."""
