"""Phase 1 CV ingestion, OCR, text cleaning, and benchmark package."""

from src.extractor import extract_directory, extract_text
from src.models import (
    CorruptedFileError,
    EncryptedPDFError,
    ExtractionError,
    ExtractionMethod,
    ExtractionResult,
    TesseractNotFoundError,
    UnsupportedFileTypeError,
)

__all__ = [
    "CorruptedFileError",
    "EncryptedPDFError",
    "ExtractionError",
    "ExtractionMethod",
    "ExtractionResult",
    "TesseractNotFoundError",
    "UnsupportedFileTypeError",
    "extract_directory",
    "extract_text",
]
