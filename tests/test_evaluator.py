"""Unit and integration tests for src.evaluator."""

import csv
from pathlib import Path

import numpy as np
import pymupdf

from src.evaluator import (
    compute_metrics,
    format_summary_table,
    load_ground_truth,
    run_benchmark,
    sample_pdfs,
    simulate_degraded_scan,
)


def _create_text_pdf(path: Path, lines: list[str]) -> None:
    """Create a simple readable text PDF for benchmark tests."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    for line in lines:
        page.insert_text((60, y), line, fontsize=14)
        y += 26
    doc.save(str(path))
    doc.close()


def test_load_ground_truth_handles_large_multiline_utf8_and_str_ids(
    tmp_path: Path,
) -> None:
    """Ground truth loader preserves string IDs, UTF-8 text, and large HTML fields."""
    csv_path = tmp_path / "Resume.csv"
    large_html = "<div>" + ("x" * 220_000) + "</div>"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "Resume_str", "Resume_html", "Category"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ID": "00123",
                "Resume_str": "Senior Engineer\nÖzgeçmiş C++",
                "Resume_html": large_html,
                "Category": "INFORMATION-TECHNOLOGY",
            }
        )

    gt = load_ground_truth(csv_path)
    assert "00123" in gt
    assert "Özgeçmiş C++" in gt["00123"]


def test_sample_pdfs_is_deterministic_with_seed(tmp_path: Path) -> None:
    """Deterministic seed selects identical matched PDFs and skips unmatched IDs."""
    gt = {f"100{i}": f"Resume text {i}" for i in range(5)}
    for resume_id in [*gt.keys(), "9999"]:
        (tmp_path / f"{resume_id}.pdf").write_bytes(b"%PDF-1.4")

    first = sample_pdfs(tmp_path, gt, n=4, seed=42)
    second = sample_pdfs(tmp_path, gt, n=4, seed=42)

    assert first == second
    assert len(first) == 4
    assert all(p.stem in gt for p in first)


def test_simulate_degraded_scan_preserves_shape_and_dtype() -> None:
    """Simulated scan degradation applies tilt + noise while preserving uint8 shape."""
    base = np.full((300, 400, 3), 255, dtype=np.uint8)
    base[100:130, 50:350] = 0
    rng = np.random.default_rng(42)

    degraded = simulate_degraded_scan(base, angle_deg=2.5, noise_std=8.0, rng=rng)
    assert degraded.shape == base.shape
    assert degraded.dtype == np.uint8
    assert not np.array_equal(degraded, base)


def test_compute_metrics_fair_normalization_and_empty_handling() -> None:
    """Metrics apply identical normalization and safely handle empty OCR strings."""
    wer_same, cer_same = compute_metrics(
        "  Senior   Engineer\n____\nC++  .NET ",
        "senior engineer c++ .net",
    )
    assert wer_same == 0.0
    assert cer_same == 0.0

    wer_diff, cer_diff = compute_metrics("python developer", "pythan developer")
    assert 0.0 < wer_diff <= 1.0
    assert 0.0 < cer_diff < 0.5

    wer_empty, cer_empty = compute_metrics("non empty reference", "")
    assert wer_empty == 1.0
    assert cer_empty == 1.0

    assert compute_metrics("", "") == (0.0, 0.0)


def test_run_benchmark_splits_clean_and_degraded_and_writes_csv(tmp_path: Path) -> None:
    """Benchmark runs clean/degraded 50-50 split, writes UTF-8 CSV, and formats summary."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    csv_path = tmp_path / "Resume.csv"
    out_csv = tmp_path / "outputs" / "ocr_benchmark.csv"

    samples = [
        ("1001", ["Software Engineer", "Python C++ Backend"]),
        ("1002", ["Data Engineer", "SQL Pipelines Docker"]),
        ("1003", ["Systems Architect", "Linux Cloud Security"]),
        ("1004", ["DevOps Engineer", "CI/CD Kubernetes Git"]),
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "Resume_str", "Resume_html", "Category"],
        )
        writer.writeheader()
        for resume_id, lines in samples:
            _create_text_pdf(pdf_dir / f"{resume_id}.pdf", lines)
            writer.writerow(
                {
                    "ID": resume_id,
                    "Resume_str": " ".join(lines),
                    "Resume_html": "",
                    "Category": "INFORMATION-TECHNOLOGY",
                }
            )

    rows = run_benchmark(
        pdf_dir=pdf_dir,
        csv_path=csv_path,
        output_csv=out_csv,
        sample_size=4,
        seed=42,
        ocr_dpi=200,
    )

    assert len(rows) == 4
    conditions = [r.condition for r in rows]
    assert conditions.count("clean") == 2
    assert conditions.count("degraded") == 2
    assert out_csv.is_file()

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 4
    assert {"resume_id", "filename", "condition", "wer", "cer"}.issubset(
        saved[0].keys()
    )

    summary = format_summary_table(rows)
    assert "clean" in summary
    assert "degraded" in summary
    assert "overall" in summary
