"""Unit and integration tests for src.extractor."""

from pathlib import Path

import docx
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from src import config
from src.extractor import extract_directory, extract_text
from src.models import (
    CorruptedFileError,
    EncryptedPDFError,
    TesseractNotFoundError,
    UnsupportedFileTypeError,
)


def _render_text_png(path: Path, lines: list[str]) -> None:
    """Render clean high-contrast text onto a PNG image for OCR tests."""
    img = Image.new("RGB", (1400, 450), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    y = 50
    for line in lines:
        draw.text((60, y), line, fill=(0, 0, 0), font=font)
        y += 80
    img.save(path)


def _create_digital_pdf(path: Path, text: str) -> None:
    """Create a selectable-text PDF using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def _create_scanned_pdf(path: Path, png_path: Path, hidden_text: str = "") -> None:
    """Create an image-only PDF (optionally with tiny hidden text under threshold)."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    if hidden_text:
        page.insert_text((10, 10), hidden_text, fontsize=6)
    page.insert_image(pymupdf.Rect(36, 60, 560, 260), filename=str(png_path))
    doc.save(str(path))
    doc.close()


def test_direct_pdf_extracts_text_with_sort_true(tmp_path: Path) -> None:
    """Digital PDF with >=50 chars routes to direct_pdf and preserves technical terms."""
    pdf_path = tmp_path / "digital_cv.pdf"
    sample_text = (
        "Senior Software Engineer\n"
        "Skills: C++, C#, .NET, Node.js, CI/CD, Python 3.12\n"
        "Built distributed backend platforms and automated pipelines."
    )
    _create_digital_pdf(pdf_path, sample_text)

    result = extract_text(pdf_path)
    assert result.method == "direct_pdf"
    assert result.page_count == 1
    assert result.filename == "digital_cv.pdf"
    assert "C++" in result.cleaned_text
    assert ".NET" in result.cleaned_text
    assert result.processing_time_sec >= 0.0


def test_scanned_pdf_below_min_text_length_routes_to_ocr_pdf(tmp_path: Path) -> None:
    """Image PDFs (even with whitespace-padded tiny text <50 chars) route to ocr_pdf."""
    png_path = tmp_path / "scan_source.png"
    _render_text_png(
        png_path,
        [
            "Software Engineer Resume",
            "Experience with Python and Docker",
        ],
    )
    pdf_path = tmp_path / "scanned_cv.pdf"
    _create_scanned_pdf(pdf_path, png_path, hidden_text=("   \n" * 30) + "CV")

    result = extract_text(pdf_path)
    assert result.method == "ocr_pdf"
    assert result.page_count == 1
    assert "Software" in result.cleaned_text
    assert "Python" in result.cleaned_text


def test_image_ocr_with_turkish_unicode_path(tmp_path: Path) -> None:
    """PNG under Turkish Unicode path routes to ocr_image without cv2 failure."""
    unicode_dir = tmp_path / "özgeçmiş_şğüöçı"
    unicode_dir.mkdir()
    img_path = unicode_dir / "belge_İ.png"
    _render_text_png(img_path, ["Backend Developer", "Python Systems"])

    result = extract_text(img_path)
    assert result.method == "ocr_image"
    assert result.page_count == 1
    assert "Backend" in result.cleaned_text
    assert "Python" in result.cleaned_text


def test_docx_extracts_paragraphs_and_table_cells_without_merged_duplication(
    tmp_path: Path,
) -> None:
    """DOCX reader extracts paragraphs and tables in order without duplicating merged cells."""
    docx_path = tmp_path / "resume_layout.docx"
    document = docx.Document()
    document.add_paragraph("Header Contact Info")
    table = document.add_table(rows=2, cols=2)
    merged = table.cell(0, 0).merge(table.cell(0, 1))
    merged.text = "Skills: C# & .NET"
    table.cell(1, 0).text = "Languages"
    table.cell(1, 1).text = "Python 3.12, C++"
    document.add_paragraph("Work Experience Summary")
    document.save(str(docx_path))

    result = extract_text(docx_path)
    assert result.method == "docx"
    assert result.page_count == 1
    assert "Header Contact Info" in result.cleaned_text
    assert "Skills: C# & .NET" in result.cleaned_text
    assert "Python 3.12, C++" in result.cleaned_text
    assert "Work Experience Summary" in result.cleaned_text
    assert result.cleaned_text.count("Skills: C# & .NET") == 1
    assert result.cleaned_text.index("Header Contact Info") < result.cleaned_text.index(
        "Skills: C# & .NET"
    ) < result.cleaned_text.index("Work Experience Summary")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    """Unsupported file extensions raise UnsupportedFileTypeError."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("plain text", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        extract_text(txt_file)


def test_corrupted_pdf_and_docx_raise(tmp_path: Path) -> None:
    """Corrupted or empty PDF and DOCX files raise CorruptedFileError."""
    bad_pdf = tmp_path / "broken.pdf"
    bad_pdf.write_bytes(b"")
    bad_docx = tmp_path / "broken.docx"
    bad_docx.write_bytes(b"not a zip archive")

    with pytest.raises(CorruptedFileError):
        extract_text(bad_pdf)
    with pytest.raises(CorruptedFileError):
        extract_text(bad_docx)


def test_encrypted_pdf_raises(tmp_path: Path) -> None:
    """Password-protected PDFs raise EncryptedPDFError."""
    enc_pdf = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Confidential CV content " * 5)
    doc.save(
        str(enc_pdf),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="owner",
    )
    doc.close()

    with pytest.raises(EncryptedPDFError):
        extract_text(enc_pdf)


def test_missing_tesseract_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing Tesseract binary raises TesseractNotFoundError on OCR routes."""
    img_path = tmp_path / "sample.png"
    _render_text_png(img_path, ["Hello OCR"])
    monkeypatch.setattr(config, "resolve_tesseract_cmd", lambda _explicit=None: None)

    with pytest.raises(TesseractNotFoundError):
        extract_text(img_path)


def test_extract_directory_processes_supported_files(tmp_path: Path) -> None:
    """Directory extraction processes supported files and ignores unsupported ones."""
    _create_digital_pdf(
        tmp_path / "a.pdf",
        "Software Engineer with extensive Python 3.12 and C++ backend experience.",
    )
    doc = docx.Document()
    doc.add_paragraph("Candidate Resume in DOCX format with Node.js skills.")
    doc.save(str(tmp_path / "b.docx"))
    (tmp_path / "ignore.txt").write_text("ignored", encoding="utf-8")

    results = extract_directory(tmp_path)
    assert [r.filename for r in results] == ["a.pdf", "b.docx"]
    assert [r.method for r in results] == ["direct_pdf", "docx"]
