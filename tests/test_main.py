"""Unit and integration tests for main.py CLI."""

import json
from pathlib import Path

import docx
import pymupdf
import pytest

from main import main


def _create_digital_pdf(path: Path, text: str) -> None:
    """Create a selectable-text PDF for CLI tests."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def test_cli_single_pdf_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Single PDF with --json prints valid ExtractionResult JSON and exits 0."""
    pdf_path = tmp_path / "candidate.pdf"
    _create_digital_pdf(
        pdf_path,
        "Full-Stack Engineer skilled in C++, C#, .NET, Node.js, CI/CD, and Python 3.12.",
    )

    exit_code = main([str(pdf_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["filename"] == "candidate.pdf"
    assert payload["method"] == "direct_pdf"
    assert "C++" in payload["cleaned_text"]


def test_cli_directory_summary_table_and_output_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Directory target writes UTF-8 JSON file with --output and renders table by default."""
    input_dir = tmp_path / "cvs"
    input_dir.mkdir()
    _create_digital_pdf(
        input_dir / "alpha.pdf",
        "Backend Software Engineer with strong Python 3.12 and PostgreSQL experience.",
    )
    doc = docx.Document()
    doc.add_paragraph("Frontend Developer experienced in Vue.js and TypeScript.")
    doc.save(str(input_dir / "beta.docx"))

    out_json = tmp_path / "result.json"
    code_json = main([str(input_dir), "--json", "--output", str(out_json)])
    assert code_json == 0
    assert out_json.is_file()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert [entry["filename"] for entry in saved] == ["alpha.pdf", "beta.docx"]

    capsys.readouterr()
    code_table = main([str(input_dir)])
    captured = capsys.readouterr()
    assert code_table == 0
    assert "alpha.pdf" in captured.out
    assert "beta.docx" in captured.out
    assert "direct_pdf" in captured.out
    assert "docx" in captured.out


def test_cli_invalid_or_missing_file_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unsupported extension and missing paths exit cleanly with non-zero code."""
    bad_txt = tmp_path / "notes.txt"
    bad_txt.write_text("hello", encoding="utf-8")

    code_unsupported = main([str(bad_txt)])
    err_unsupported = capsys.readouterr().err
    assert code_unsupported != 0
    assert "Unsupported" in err_unsupported

    code_missing = main([str(tmp_path / "does_not_exist.pdf")])
    err_missing = capsys.readouterr().err
    assert code_missing != 0
    assert "not found" in err_missing.lower()
