"""Unit and integration tests for src.ui_helpers."""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.models import CorruptedFileError
from src.ui_helpers import (
    add_items_to_workspace,
    compute_cleaning_stats,
    compute_content_hash,
    decode_image_bytes,
    export_workspace_csv,
    export_workspace_json,
    extract_uploaded_bytes,
    find_sample_cv,
    load_benchmark_summary,
    remove_workspace_item,
    render_pdf_preview,
    render_preprocessed_preview,
    reprocess_workspace,
)


def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Create an in-memory digital PDF with one or more text pages."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=12)
    payload = doc.tobytes()
    doc.close()
    return payload


def _make_png_bytes(lines: list[str]) -> bytes:
    """Render high-contrast text to PNG bytes."""
    img = Image.new("RGB", (1200, 360), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        font = ImageFont.load_default(size=44)
    y = 40
    for line in lines:
        draw.text((50, y), line, fill=(0, 0, 0), font=font)
        y += 75
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_extract_uploaded_bytes_unicode_filename_and_temp_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uploaded bytes with Turkish Unicode filename extract cleanly and leave zero temp files."""
    isolated_tmp = tmp_path / "temp_root"
    isolated_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(isolated_tmp))

    png_bytes = _make_png_bytes(["Senior Developer", "Python Systems"])
    result = extract_uploaded_bytes("Özgeçmiş_Şükrü_İ.png", png_bytes)

    assert result.filename == "Özgeçmiş_Şükrü_İ.png"
    assert result.file_path == "Özgeçmiş_Şükrü_İ.png"
    assert result.method == "ocr_image"
    assert "Senior" in result.cleaned_text
    assert list(isolated_tmp.iterdir()) == []

    with pytest.raises(CorruptedFileError):
        extract_uploaded_bytes("broken_özgeçmiş.pdf", b"not-a-pdf")
    assert list(isolated_tmp.iterdir()) == []


def test_additive_workspace_queue_and_sha256_deduplication() -> None:
    """Subsequent batches append without resetting and deduplicate strictly by SHA-256."""
    pdf_a = _make_pdf_bytes(
        ["Candidate Alpha Software Engineer skilled in C++, C#, .NET, and Python 3.12."]
    )
    pdf_b = _make_pdf_bytes(
        ["Candidate Beta Data Engineer experienced with PL/SQL, Node.js, and CI/CD."]
    )
    pdf_c = _make_pdf_bytes(
        ["Candidate Gamma Cloud Architect building TCP/IP and Vue.js platforms."]
    )

    ws, added_1, dup_1, err_1 = add_items_to_workspace([], [("alpha.pdf", pdf_a)])
    assert len(ws) == 1
    assert added_1 == ["alpha.pdf"]
    assert dup_1 == []
    assert err_1 == []

    ws, added_2, dup_2, err_2 = add_items_to_workspace(
        ws,
        [
            ("beta.pdf", pdf_b),
            ("alpha_copy.pdf", pdf_a),
            ("gamma.pdf", pdf_c),
        ],
    )
    assert len(ws) == 3
    assert [item.filename for item in ws] == ["alpha.pdf", "beta.pdf", "gamma.pdf"]
    assert added_2 == ["beta.pdf", "gamma.pdf"]
    assert dup_2 == ["alpha_copy.pdf"]
    assert err_2 == []
    assert ws[0].content_hash == compute_content_hash(pdf_a)


def test_additive_workspace_partial_failure_keeps_valid_files() -> None:
    """Corrupted or unsupported files report errors without dropping valid batch items."""
    valid_pdf = _make_pdf_bytes(
        ["Valid Resume Software Engineer specializing in Python 3.12 backend services."]
    )
    ws, added, dups, errors = add_items_to_workspace(
        [],
        [
            ("broken.pdf", b""),
            ("valid.pdf", valid_pdf),
            ("notes.txt", b"unsupported plain text"),
        ],
    )
    assert len(ws) == 1
    assert added == ["valid.pdf"]
    assert dups == []
    assert len(errors) == 2


def test_remove_and_reprocess_workspace() -> None:
    """Removing by hash drops only the target item; reprocessing updates settings."""
    pdf_a = _make_pdf_bytes(
        ["Candidate Alpha Software Engineer with extensive Python 3.12 experience."]
    )
    pdf_b = _make_pdf_bytes(
        ["Candidate Beta Systems Engineer with C++ and Linux kernel background."]
    )
    ws, _, _, _ = add_items_to_workspace(
        [],
        [("a.pdf", pdf_a), ("b.pdf", pdf_b)],
        ocr_dpi=300,
        apply_threshold=False,
    )

    remaining = remove_workspace_item(ws, ws[0].content_hash)
    assert [item.filename for item in remaining] == ["b.pdf"]

    reprocessed, errors = reprocess_workspace(
        remaining,
        ocr_dpi=200,
        apply_threshold=True,
    )
    assert errors == []
    assert len(reprocessed) == 1
    assert reprocessed[0].ocr_dpi == 200
    assert reprocessed[0].apply_threshold is True
    assert "Candidate Beta" in reprocessed[0].result.cleaned_text


def test_render_pdf_preview_multipage_and_clamping() -> None:
    """PDF preview renders requested page as RGB uint8 and clamps out-of-range indices."""
    pdf_bytes = _make_pdf_bytes(["Page One Content", "Page Two Content"])

    img_0, count_0 = render_pdf_preview(pdf_bytes, page_index=0, preview_dpi=100)
    img_1, count_1 = render_pdf_preview(pdf_bytes, page_index=1, preview_dpi=100)
    img_clamped, count_c = render_pdf_preview(pdf_bytes, page_index=99, preview_dpi=100)

    assert count_0 == count_1 == count_c == 2
    assert img_0.ndim == 3 and img_0.shape[2] == 3 and img_0.dtype == np.uint8
    assert np.array_equal(img_1, img_clamped)


def test_decode_image_bytes_and_preprocessed_preview() -> None:
    """Image bytes decode to RGB, fail cleanly on garbage, and produce OpenCV previews."""
    png_bytes = _make_png_bytes(["OpenCV Preview Test"])
    rgb = decode_image_bytes(png_bytes)
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8

    with pytest.raises(CorruptedFileError):
        decode_image_bytes(b"corrupted-image-payload")

    gray_preview = render_preprocessed_preview(rgb, apply_threshold=False)
    bin_preview = render_preprocessed_preview(rgb, apply_threshold=True)
    assert gray_preview.ndim == 2 and gray_preview.dtype == np.uint8
    assert bin_preview.ndim == 2 and set(np.unique(bin_preview)).issubset({0, 255})


def test_compute_cleaning_stats_and_tech_term_detection() -> None:
    """Cleaning stats measure noise reduction accurately and detect preserved tech terms."""
    raw = (
        "____\n"
        "Skills: C++, C#, .NET, ASP.NET, Node.js, Vue.js, CI/CD, TCP/IP, PL/SQL\n"
        "======"
    )
    cleaned = "Skills: C++, C#, .NET, ASP.NET, Node.js, Vue.js, CI/CD, TCP/IP, PL/SQL"
    stats = compute_cleaning_stats(raw, cleaned)

    assert stats.raw_chars == len(raw)
    assert stats.cleaned_chars == len(cleaned)
    assert stats.removed_chars == len(raw) - len(cleaned)
    assert stats.reduction_pct > 0.0
    assert stats.preserved_tech_terms == [
        "C++",
        "C#",
        ".NET",
        "ASP.NET",
        "Node.js",
        "Vue.js",
        "CI/CD",
        "TCP/IP",
        "PL/SQL",
    ]

    empty_stats = compute_cleaning_stats("", "")
    assert empty_stats.reduction_pct == 0.0
    assert empty_stats.preserved_tech_terms == []


def test_export_workspace_json_and_csv() -> None:
    """Workspace exports valid UTF-8 JSON array and properly headed summary CSV."""
    pdf_bytes = _make_pdf_bytes(
        ["Engineer Özgeçmiş with Python 3.12, C++, and distributed systems skills."]
    )
    ws, _, _, _ = add_items_to_workspace([], [("özgeçmiş.pdf", pdf_bytes)])

    json_str = export_workspace_json(ws)
    parsed = json.loads(json_str)
    assert len(parsed) == 1
    assert parsed[0]["filename"] == "özgeçmiş.pdf"
    assert parsed[0]["method"] == "direct_pdf"

    csv_str = export_workspace_csv(ws)
    rows = list(csv.DictReader(io.StringIO(csv_str)))
    assert len(rows) == 1
    assert list(rows[0].keys()) == [
        "Filename",
        "Method",
        "Pages",
        "Cleaned Chars",
        "Time (s)",
        "Warnings",
    ]
    assert rows[0]["Filename"] == "özgeçmiş.pdf"
    assert rows[0]["Method"] == "direct_pdf"


def test_find_sample_cv_present_and_missing(tmp_path: Path) -> None:
    """Sample CV finder returns first sorted PDF if present, None if missing or empty."""
    assert find_sample_cv(tmp_path / "nonexistent") is None
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert find_sample_cv(empty_dir) is None

    (empty_dir / "200.pdf").write_bytes(b"%PDF-1.4")
    (empty_dir / "100.pdf").write_bytes(b"%PDF-1.4")
    assert find_sample_cv(empty_dir) == empty_dir / "100.pdf"


def test_load_benchmark_summary_parses_phase1_csv_and_handles_missing(
    tmp_path: Path,
) -> None:
    """Benchmark loader handles missing CSV and aggregates clean/degraded/overall metrics."""
    assert load_benchmark_summary(tmp_path / "missing.csv") is None

    csv_path = tmp_path / "ocr_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "resume_id",
                "filename",
                "condition",
                "page_count",
                "wer",
                "cer",
                "processing_time_sec",
                "gt_char_len",
                "ocr_char_len",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "resume_id": "1001",
                "filename": "1001.pdf",
                "condition": "clean",
                "page_count": 2,
                "wer": 0.02,
                "cer": 0.01,
                "processing_time_sec": 3.0,
                "gt_char_len": 1000,
                "ocr_char_len": 1005,
            }
        )
        writer.writerow(
            {
                "resume_id": "1002",
                "filename": "1002.pdf",
                "condition": "degraded",
                "page_count": 2,
                "wer": 0.06,
                "cer": 0.03,
                "processing_time_sec": 5.0,
                "gt_char_len": 1200,
                "ocr_char_len": 1190,
            }
        )

    summary = load_benchmark_summary(csv_path)
    assert summary is not None
    assert summary.total_samples == 2
    assert summary.overall_wer == pytest.approx(0.04)
    assert summary.overall_cer == pytest.approx(0.02)
    assert summary.avg_time_sec == pytest.approx(4.0)
    assert summary.group_stats["clean"]["mean_wer"] == pytest.approx(0.02)
    assert summary.group_stats["degraded"]["mean_wer"] == pytest.approx(0.06)
    assert summary.rows[0]["ID"] == "1001"
    assert summary.rows[0]["Group"] == "clean"
