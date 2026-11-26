"""Unit tests for src.config and src.models."""

from pathlib import Path

from src import config
from src.models import (
    CorruptedFileError,
    EncryptedPDFError,
    ExtractionError,
    ExtractionResult,
    TesseractNotFoundError,
    UnsupportedFileTypeError,
)


def test_config_spec_constants() -> None:
    """Config exposes all spec-pinned constants and pathlib paths."""
    assert config.MIN_TEXT_LENGTH == 50
    assert config.OCR_DPI == 300
    assert config.TESSERACT_LANG == "eng"
    assert config.TESSERACT_CONFIG == "--oem 3 --psm 3"
    assert config.RANDOM_SEED == 42
    assert config.DESKEW_MIN_ANGLE == 0.5
    assert config.DESKEW_MAX_ANGLE == 15.0
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.SUPPORTED_EXTENSIONS == {".pdf", ".png", ".jpg", ".jpeg", ".docx"}


def test_resolve_tesseract_cmd_prefers_explicit_and_finds_system(tmp_path: Path) -> None:
    """Explicit executable path takes precedence; system PATH fallback resolves."""
    fake_exe = tmp_path / "tesseract.exe"
    fake_exe.write_text("stub", encoding="utf-8")
    assert config.resolve_tesseract_cmd(str(fake_exe)) == str(fake_exe)
    assert config.resolve_tesseract_cmd() is not None


def test_extraction_result_schema_and_to_dict() -> None:
    """ExtractionResult exposes the exact Phase 2 schema and serializes cleanly."""
    result = ExtractionResult(
        file_path="C:/tmp/cv.pdf",
        filename="cv.pdf",
        method="direct_pdf",
        page_count=2,
        raw_text="Raw  text",
        cleaned_text="Raw text",
        processing_time_sec=0.12,
        warnings=["minor"],
    )
    payload = result.to_dict()
    assert set(payload.keys()) == {
        "file_path",
        "filename",
        "method",
        "page_count",
        "raw_text",
        "cleaned_text",
        "processing_time_sec",
        "warnings",
    }
    assert payload["method"] == "direct_pdf"
    assert payload["warnings"] == ["minor"]


def test_exception_hierarchy() -> None:
    """All domain exceptions inherit from ExtractionError."""
    for exc_cls in (
        UnsupportedFileTypeError,
        CorruptedFileError,
        EncryptedPDFError,
        TesseractNotFoundError,
    ):
        assert issubclass(exc_cls, ExtractionError)
