"""Smart-routing CV text extractor supporting PDF, image, and DOCX documents."""

from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import docx
import numpy as np
import pymupdf
import pytesseract
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from src import config
from src.cleaner import clean_text
from src.models import (
    CorruptedFileError,
    EncryptedPDFError,
    ExtractionMethod,
    ExtractionResult,
    TesseractNotFoundError,
    UnsupportedFileTypeError,
)
from src.preprocessor import load_image_unicode_safe, preprocess_image

logger = logging.getLogger(__name__)


def ensure_tesseract_available(tesseract_cmd: str | None = None) -> str:
    """Resolve and configure the Tesseract OCR binary for ``pytesseract``.

    Args:
        tesseract_cmd: Optional explicit executable path override.

    Returns:
        Resolved path to the Tesseract executable.

    Raises:
        TesseractNotFoundError: If Tesseract cannot be located on the system.
    """
    resolved = config.resolve_tesseract_cmd(tesseract_cmd)
    if not resolved:
        raise TesseractNotFoundError(
            "Tesseract OCR executable not found. Install Tesseract 5, add it to PATH, "
            "or set the TESSERACT_CMD environment variable."
        )
    pytesseract.pytesseract.tesseract_cmd = resolved
    return resolved


def _ocr_image_array(
    image: np.ndarray,
    *,
    apply_threshold: bool,
    tesseract_lang: str,
    tesseract_config: str,
) -> str:
    """Preprocess an image array and extract text using Tesseract OCR."""
    processed = preprocess_image(
        image,
        apply_deskew=True,
        apply_threshold=apply_threshold,
    )
    try:
        return str(
            pytesseract.image_to_string(
                processed,
                lang=tesseract_lang,
                config=tesseract_config,
            )
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractNotFoundError(
            "Tesseract OCR failed to execute. Verify Tesseract 5 installation."
        ) from exc


def _extract_pdf(
    path: Path,
    *,
    min_text_length: int,
    ocr_dpi: int,
    apply_threshold: bool,
    tesseract_lang: str,
    tesseract_config: str,
    tesseract_cmd: str | None,
) -> tuple[ExtractionMethod, int, str, list[str]]:
    """Extract text from a PDF using direct text layer or OCR fallback."""
    warnings: list[str] = []
    try:
        doc = pymupdf.open(str(path))
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise CorruptedFileError(f"Failed to open PDF file '{path}': {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise EncryptedPDFError(
                f"PDF file '{path}' is password-protected and cannot be read without credentials."
            )

        page_count = doc.page_count
        if page_count == 0:
            raise CorruptedFileError(f"PDF file '{path}' contains zero pages.")

        direct_pages = [page.get_text("text", sort=True) for page in doc]
        raw_direct = "\n".join(direct_pages)
        non_whitespace_len = len("".join(raw_direct.split()))

        if non_whitespace_len >= min_text_length:
            logger.info(
                "Routed '%s' to direct_pdf (%d pages, %d non-ws chars).",
                path.name,
                page_count,
                non_whitespace_len,
            )
            return "direct_pdf", page_count, raw_direct, warnings

        logger.info(
            "Routed '%s' to ocr_pdf (%d non-ws chars < threshold %d).",
            path.name,
            non_whitespace_len,
            min_text_length,
        )
        ensure_tesseract_available(tesseract_cmd)

        ocr_pages: list[str] = []
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(dpi=ocr_dpi, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height,
                pix.width,
                pix.n,
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            page_text = _ocr_image_array(
                bgr,
                apply_threshold=apply_threshold,
                tesseract_lang=tesseract_lang,
                tesseract_config=tesseract_config,
            )
            if not page_text.strip():
                warnings.append(f"Page {idx + 1} produced empty OCR text.")
            ocr_pages.append(page_text)

        return "ocr_pdf", page_count, "\n".join(ocr_pages), warnings


def _extract_image(
    path: Path,
    *,
    apply_threshold: bool,
    tesseract_lang: str,
    tesseract_config: str,
    tesseract_cmd: str | None,
) -> tuple[ExtractionMethod, int, str, list[str]]:
    """Extract text from a raster image file via OpenCV preprocessing + OCR."""
    ensure_tesseract_available(tesseract_cmd)
    image = load_image_unicode_safe(path)
    raw_text = _ocr_image_array(
        image,
        apply_threshold=apply_threshold,
        tesseract_lang=tesseract_lang,
        tesseract_config=tesseract_config,
    )
    warnings: list[str] = []
    if not raw_text.strip():
        warnings.append("OCR produced empty text for image.")
    logger.info("Routed '%s' to ocr_image.", path.name)
    return "ocr_image", 1, raw_text, warnings


def _collect_docx_blocks(container: Any, seen_tcs: set[int]) -> list[str]:
    """Recursively collect paragraphs and table cell texts in document order."""
    blocks: list[str] = []
    for item in container.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                blocks.append(text)
        elif isinstance(item, Table):
            for row in item.rows:
                row_values: list[str] = []
                for cell in row.cells:
                    tc_id = id(cell._tc)
                    if tc_id in seen_tcs:
                        continue
                    seen_tcs.add(tc_id)
                    nested = _collect_docx_blocks(cell, seen_tcs)
                    if nested:
                        row_values.append(" ".join(nested))
                if row_values:
                    blocks.append(" | ".join(row_values))
    return blocks


def _extract_docx(path: Path) -> tuple[ExtractionMethod, int, str, list[str]]:
    """Extract paragraphs and table contents in reading order from a DOCX file."""
    try:
        document = docx.Document(str(path))
    except (PackageNotFoundError, zipfile.BadZipFile, ValueError, KeyError) as exc:
        raise CorruptedFileError(f"Failed to open DOCX file '{path}': {exc}") from exc

    blocks = _collect_docx_blocks(document, set())
    raw_text = "\n".join(blocks)
    warnings: list[str] = []
    if not raw_text.strip():
        warnings.append("DOCX document contained no readable text.")
    logger.info("Routed '%s' to docx.", path.name)
    return "docx", 1, raw_text, warnings


def extract_text(
    file_path: Path | str,
    *,
    min_text_length: int = config.MIN_TEXT_LENGTH,
    ocr_dpi: int = config.OCR_DPI,
    apply_threshold: bool = False,
    tesseract_lang: str = config.TESSERACT_LANG,
    tesseract_config: str = config.TESSERACT_CONFIG,
    tesseract_cmd: str | None = None,
) -> ExtractionResult:
    """Route a CV file to the optimal extraction pipeline and return structured text.

    Supports digital and scanned PDFs (via PyMuPDF), PNG/JPG/JPEG images (via
    Unicode-safe OpenCV + Tesseract 5), and Word ``.docx`` files (paragraphs and
    tables in reading order).

    Args:
        file_path: Path to the target CV file.
        min_text_length: Non-whitespace character threshold for PDF OCR fallback.
        ocr_dpi: Resolution in DPI when rasterizing scanned PDF pages.
        apply_threshold: Whether to enable adaptive thresholding in preprocessing.
        tesseract_lang: Tesseract language pack identifier.
        tesseract_config: Extra Tesseract CLI configuration flags.
        tesseract_cmd: Optional explicit path to the Tesseract executable.

    Returns:
        Populated :class:`ExtractionResult` containing both ``raw_text`` and
        ``cleaned_text``.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        UnsupportedFileTypeError: If the file extension is not supported.
        CorruptedFileError: If the file is unreadable or malformed.
        EncryptedPDFError: If the PDF requires a password.
        TesseractNotFoundError: If OCR is required but Tesseract is unavailable.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{suffix}' for '{path.name}'. "
            f"Supported extensions: {supported}."
        )

    start_time = time.perf_counter()

    if suffix in config.SUPPORTED_PDF_EXTENSIONS:
        method, page_count, raw_text, warnings = _extract_pdf(
            path,
            min_text_length=min_text_length,
            ocr_dpi=ocr_dpi,
            apply_threshold=apply_threshold,
            tesseract_lang=tesseract_lang,
            tesseract_config=tesseract_config,
            tesseract_cmd=tesseract_cmd,
        )
    elif suffix in config.SUPPORTED_IMAGE_EXTENSIONS:
        method, page_count, raw_text, warnings = _extract_image(
            path,
            apply_threshold=apply_threshold,
            tesseract_lang=tesseract_lang,
            tesseract_config=tesseract_config,
            tesseract_cmd=tesseract_cmd,
        )
    else:
        method, page_count, raw_text, warnings = _extract_docx(path)

    cleaned_text = clean_text(raw_text)
    elapsed = round(time.perf_counter() - start_time, 4)

    return ExtractionResult(
        file_path=str(path.resolve()),
        filename=path.name,
        method=method,
        page_count=page_count,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        processing_time_sec=elapsed,
        warnings=warnings,
    )


def extract_directory(
    dir_path: Path | str,
    *,
    recursive: bool = False,
    **kwargs: Any,
) -> list[ExtractionResult]:
    """Extract text from all supported CV files in a directory.

    Unsupported files in the directory are skipped; supported files are processed
    in deterministic sorted order.

    Args:
        dir_path: Directory containing CV documents.
        recursive: Whether to traverse subdirectories recursively.
        **kwargs: Additional keyword arguments forwarded to :func:`extract_text`.

    Returns:
        List of :class:`ExtractionResult` objects in sorted path order.

    Raises:
        FileNotFoundError: If ``dir_path`` does not exist.
        NotADirectoryError: If ``dir_path`` is not a directory.
    """
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    files = sorted(
        (
            candidate
            for candidate in iterator
            if candidate.is_file()
            and candidate.suffix.lower() in config.SUPPORTED_EXTENSIONS
        ),
        key=lambda p: str(p).lower(),
    )

    results: list[ExtractionResult] = []
    for file_path in files:
        results.append(extract_text(file_path, **kwargs))
    return results
