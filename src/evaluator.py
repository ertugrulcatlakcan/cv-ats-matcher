"""Ground-truth OCR stress test and benchmark evaluator using Resume.csv and jiwer."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import jiwer
import numpy as np
import pymupdf
import pytesseract

from src import config
from src.cleaner import clean_text, normalize_for_comparison
from src.extractor import ensure_tesseract_available
from src.preprocessor import preprocess_image

logger = logging.getLogger(__name__)

ScanCondition = Literal["clean", "degraded"]

_CSV_FIELD_LIMIT = min(sys.maxsize, 2**31 - 1)


@dataclass
class BenchmarkRow:
    """Single evaluated resume entry in the OCR benchmark report.

    Attributes:
        resume_id: String ID matched against ``Resume.csv``.
        filename: Source PDF filename.
        condition: Evaluation condition (``clean`` or ``degraded``).
        page_count: Number of pages rendered and OCR'd.
        wer: Word Error Rate computed after fair normalization.
        cer: Character Error Rate computed after fair normalization.
        processing_time_sec: Elapsed OCR + cleaning time in seconds.
        gt_char_len: Character length of normalized ground truth text.
        ocr_char_len: Character length of normalized OCR text.
    """

    resume_id: str
    filename: str
    condition: ScanCondition
    page_count: int
    wer: float
    cer: float
    processing_time_sec: float
    gt_char_len: int
    ocr_char_len: int


def load_ground_truth(csv_path: Path = config.RESUME_CSV_PATH) -> dict[str, str]:
    """Load ``ID -> Resume_str`` mappings from ``Resume.csv`` using UTF-8.

    Safely raises Python's CSV field size limit for large multiline HTML columns
    and preserves IDs strictly as strings.

    Args:
        csv_path: Path to ``Resume.csv``.

    Returns:
        Dictionary mapping string resume IDs to raw ground-truth ``Resume_str`` text.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth CSV not found: {path}")

    csv.field_size_limit(_CSV_FIELD_LIMIT)
    ground_truth: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            resume_id = str(row.get("ID", "")).strip()
            if resume_id:
                ground_truth[resume_id] = row.get("Resume_str") or ""

    logger.info("Loaded %d ground-truth resumes from '%s'.", len(ground_truth), path)
    return ground_truth


def sample_pdfs(
    pdf_dir: Path,
    ground_truth: dict[str, str],
    n: int = 10,
    seed: int = config.RANDOM_SEED,
) -> list[Path]:
    """Deterministically sample PDFs whose stem ID exists in ground truth.

    Args:
        pdf_dir: Directory containing ``<ID>.pdf`` files.
        ground_truth: Mapping of string resume IDs to ground truth text.
        n: Maximum number of PDFs to sample.
        seed: Random seed for reproducible selection.

    Returns:
        Deterministic list of matched PDF paths.

    Raises:
        FileNotFoundError: If ``pdf_dir`` does not exist.
        ValueError: If no PDFs match any ground-truth ID.
    """
    directory = Path(pdf_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {directory}")

    matched = sorted(
        (
            pdf_path
            for pdf_path in directory.glob("*.pdf")
            if pdf_path.is_file() and pdf_path.stem in ground_truth
        ),
        key=lambda p: p.name,
    )
    if not matched:
        raise ValueError(
            f"No PDF files in '{directory}' matched any ID in ground truth CSV."
        )

    count = min(max(1, n), len(matched))
    rng = random.Random(seed)
    return rng.sample(matched, count)


def simulate_degraded_scan(
    image: np.ndarray,
    *,
    angle_deg: float,
    noise_std: float = 8.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply realistic scan degradation (tilt rotation and mild Gaussian noise).

    Args:
        image: Input uint8 BGR/RGB image array.
        angle_deg: Rotation angle in degrees (typically +/-2 to +/-3 degrees).
        noise_std: Standard deviation of additive zero-mean Gaussian noise.
        rng: NumPy random generator for reproducibility.

    Returns:
        Degraded uint8 image array with identical shape to ``image``.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    border_val: int | tuple[int, int, int] = (255, 255, 255) if image.ndim == 3 else 255
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )

    noise = rng.normal(loc=0.0, scale=noise_std, size=rotated.shape)
    noisy = np.clip(rotated.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return noisy


def compute_metrics(ground_truth_text: str, ocr_text: str) -> tuple[float, float]:
    """Compute fair Word Error Rate (WER) and Character Error Rate (CER).

    Both strings pass through :func:`src.cleaner.normalize_for_comparison` before
    evaluation so formatting noise does not bias metrics.

    Args:
        ground_truth_text: Reference text from ``Resume.csv``.
        ocr_text: Hypothesis text produced by OCR + cleaner.

    Returns:
        Tuple of ``(wer, cer)`` as floats.
    """
    ref = normalize_for_comparison(ground_truth_text)
    hyp = normalize_for_comparison(ocr_text)

    if not ref and not hyp:
        return 0.0, 0.0
    if not ref or not hyp:
        return 1.0, 1.0

    return float(jiwer.wer(ref, hyp)), float(jiwer.cer(ref, hyp))


def _ocr_pdf_as_images(
    pdf_path: Path,
    *,
    condition: ScanCondition,
    ocr_dpi: int,
    rng: np.random.Generator,
    tesseract_lang: str,
    tesseract_config: str,
) -> tuple[int, str]:
    """Rasterize PDF pages to destroy the text layer, optionally degrade, and OCR."""
    ocr_pages: list[str] = []
    with pymupdf.open(str(pdf_path)) as doc:
        page_count = doc.page_count
        for page in doc:
            pix = page.get_pixmap(dpi=ocr_dpi, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height,
                pix.width,
                pix.n,
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if condition == "degraded":
                sign = -1.0 if float(rng.random()) < 0.5 else 1.0
                angle = sign * float(rng.uniform(2.0, 3.0))
                bgr = simulate_degraded_scan(bgr, angle_deg=angle, noise_std=8.0, rng=rng)

            processed = preprocess_image(bgr, apply_deskew=True, apply_threshold=False)
            text = str(
                pytesseract.image_to_string(
                    processed,
                    lang=tesseract_lang,
                    config=tesseract_config,
                )
            )
            ocr_pages.append(text)

    return page_count, clean_text("\n".join(ocr_pages))


def run_benchmark(
    pdf_dir: Path = config.DEFAULT_BENCHMARK_DIR,
    csv_path: Path = config.RESUME_CSV_PATH,
    output_csv: Path = config.BENCHMARK_OUTPUT_PATH,
    sample_size: int = 10,
    seed: int = config.RANDOM_SEED,
    ocr_dpi: int = config.OCR_DPI,
    tesseract_lang: str = config.TESSERACT_LANG,
    tesseract_config: str = config.TESSERACT_CONFIG,
) -> list[BenchmarkRow]:
    """Run the deterministic clean vs. degraded OCR benchmark against Resume.csv.

    Selects ``sample_size`` PDFs with ``seed``, rasterizes every page to remove
    embedded PDF text, evaluates the first half as clean scans and the second
    half with simulated tilt + Gaussian noise, and writes per-document metrics
    to ``output_csv`` using UTF-8.

    Args:
        pdf_dir: Directory containing source ``<ID>.pdf`` resumes.
        csv_path: Path to ``Resume.csv`` ground truth file.
        output_csv: Destination CSV path for detailed benchmark rows.
        sample_size: Number of PDFs to benchmark (default 10).
        seed: Seed controlling sampling and synthetic scan degradation.
        ocr_dpi: Rasterization DPI passed to PyMuPDF.
        tesseract_lang: Tesseract language pack identifier.
        tesseract_config: Tesseract CLI configuration string.

    Returns:
        List of :class:`BenchmarkRow` results.
    """
    ensure_tesseract_available()
    ground_truth = load_ground_truth(csv_path)
    selected_pdfs = sample_pdfs(pdf_dir, ground_truth, n=sample_size, seed=seed)

    clean_count = len(selected_pdfs) // 2
    rng = np.random.default_rng(seed)
    rows: list[BenchmarkRow] = []

    for idx, pdf_path in enumerate(selected_pdfs):
        condition: ScanCondition = "clean" if idx < clean_count else "degraded"
        resume_id = pdf_path.stem
        gt_text = ground_truth[resume_id]

        start = time.perf_counter()
        page_count, ocr_cleaned = _ocr_pdf_as_images(
            pdf_path,
            condition=condition,
            ocr_dpi=ocr_dpi,
            rng=rng,
            tesseract_lang=tesseract_lang,
            tesseract_config=tesseract_config,
        )
        elapsed = round(time.perf_counter() - start, 4)

        wer_score, cer_score = compute_metrics(gt_text, ocr_cleaned)
        row = BenchmarkRow(
            resume_id=resume_id,
            filename=pdf_path.name,
            condition=condition,
            page_count=page_count,
            wer=round(wer_score, 4),
            cer=round(cer_score, 4),
            processing_time_sec=elapsed,
            gt_char_len=len(normalize_for_comparison(gt_text)),
            ocr_char_len=len(normalize_for_comparison(ocr_cleaned)),
        )
        rows.append(row)
        logger.info(
            "[%d/%d] %s (%s): WER=%.4f CER=%.4f (%.2fs)",
            idx + 1,
            len(selected_pdfs),
            pdf_path.name,
            condition,
            row.wer,
            row.cer,
            row.processing_time_sec,
        )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkRow.__dataclass_fields__.keys())
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    logger.info("Saved benchmark report to '%s'.", out_path)
    return rows


def format_summary_table(rows: list[BenchmarkRow]) -> str:
    """Format an ASCII summary table grouped by clean, degraded, and overall results.

    Args:
        rows: Benchmark rows produced by :func:`run_benchmark`.

    Returns:
        Formatted multi-line summary table string.
    """
    header = (
        f"{'Group':<12} | {'Count':>5} | {'Mean WER':>10} | "
        f"{'Mean CER':>10} | {'Mean Time (s)':>13}"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    groups: list[tuple[str, list[BenchmarkRow]]] = [
        ("clean", [r for r in rows if r.condition == "clean"]),
        ("degraded", [r for r in rows if r.condition == "degraded"]),
        ("overall", list(rows)),
    ]

    for label, group_rows in groups:
        if not group_rows:
            lines.append(
                f"{label:<12} | {0:>5} | {'N/A':>10} | {'N/A':>10} | {'N/A':>13}"
            )
            continue
        mean_wer = sum(r.wer for r in group_rows) / len(group_rows)
        mean_cer = sum(r.cer for r in group_rows) / len(group_rows)
        mean_time = sum(r.processing_time_sec for r in group_rows) / len(group_rows)
        lines.append(
            f"{label:<12} | {len(group_rows):>5} | {mean_wer:>10.4f} | "
            f"{mean_cer:>10.4f} | {mean_time:>13.2f}"
        )

    return "\n".join(lines)


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build argument parser for standalone evaluator execution."""
    parser = argparse.ArgumentParser(
        description="Run OCR stress test and WER/CER benchmark against Resume.csv."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=config.DEFAULT_BENCHMARK_DIR,
        help="Directory containing <ID>.pdf resumes.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=config.RESUME_CSV_PATH,
        help="Path to Resume.csv ground truth file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.BENCHMARK_OUTPUT_PATH,
        help="Destination path for ocr_benchmark.csv.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of PDFs to sample deterministically (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for sampling and degradation (default: 42).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=config.OCR_DPI,
        help="PDF rasterization DPI for OCR benchmark (default: 300).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def _cli_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m src.evaluator``."""
    args = _build_cli_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    rows = run_benchmark(
        pdf_dir=args.pdf_dir,
        csv_path=args.csv_path,
        output_csv=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        ocr_dpi=args.dpi,
    )
    sys.stdout.write(format_summary_table(rows) + "\n")
    sys.stdout.write(f"Saved detailed benchmark CSV to: {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
