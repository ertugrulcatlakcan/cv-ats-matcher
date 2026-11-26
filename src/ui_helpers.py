"""Streamlit-independent helper layer for Phase 1.5 ATS Web UI."""

from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf

from src import config
from src.extractor import extract_text
from src.models import CorruptedFileError, ExtractionError, ExtractionResult
from src.preprocessor import preprocess_image

logger = logging.getLogger(__name__)

TRACKED_TECH_TERMS: tuple[str, ...] = (
    "C++",
    "C#",
    ".NET",
    "ASP.NET",
    "Node.js",
    "Vue.js",
    "CI/CD",
    "TCP/IP",
    "PL/SQL",
)

_TECH_TERM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("C++", re.compile(r"(?<![A-Za-z0-9_])C\+\+(?![A-Za-z0-9_+])")),
    ("C#", re.compile(r"(?<![A-Za-z0-9_])C#(?![A-Za-z0-9_#])")),
    (".NET", re.compile(r"(?<![A-Za-z0-9_.])\.NET(?![A-Za-z0-9_])")),
    ("ASP.NET", re.compile(r"(?<![A-Za-z0-9_])ASP\.NET(?![A-Za-z0-9_])")),
    ("Node.js", re.compile(r"(?<![A-Za-z0-9_])Node\.js(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("Vue.js", re.compile(r"(?<![A-Za-z0-9_])Vue\.js(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("CI/CD", re.compile(r"(?<![A-Za-z0-9_])CI/CD(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("TCP/IP", re.compile(r"(?<![A-Za-z0-9_])TCP/IP(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("PL/SQL", re.compile(r"(?<![A-Za-z0-9_])PL/SQL(?![A-Za-z0-9_])", re.IGNORECASE)),
)


@dataclass
class WorkspaceItem:
    """Single uploaded CV record stored in the additive UI workspace.

    Attributes:
        content_hash: SHA-256 hex digest of ``file_bytes`` used for deduplication.
        filename: Original sanitized filename.
        extension: Lowercase file suffix (e.g., ``.pdf``, ``.png``, ``.docx``).
        file_bytes: Raw uploaded file bytes retained for preview and reprocessing.
        size_bytes: Payload size in bytes.
        result: Extraction output returned by Phase 1 ``extract_text``.
        ocr_dpi: DPI parameter used when extracting this item.
        apply_threshold: Adaptive thresholding toggle used when extracting this item.
    """

    content_hash: str
    filename: str
    extension: str
    file_bytes: bytes
    size_bytes: int
    result: ExtractionResult
    ocr_dpi: int
    apply_threshold: bool


@dataclass
class CleaningStats:
    """Comparison statistics between raw and cleaned extraction text.

    Attributes:
        raw_chars: Character count of ``raw_text``.
        cleaned_chars: Character count of ``cleaned_text``.
        removed_chars: Number of characters removed during cleaning.
        reduction_pct: Percentage of characters removed (0.0 when raw text is empty).
        preserved_tech_terms: Canonical technical terms preserved in ``cleaned_text``.
    """

    raw_chars: int
    cleaned_chars: int
    removed_chars: int
    reduction_pct: float
    preserved_tech_terms: list[str]


@dataclass
class BenchmarkSummary:
    """Aggregated summary parsed from ``outputs/ocr_benchmark.csv``.

    Attributes:
        total_samples: Number of evaluated resumes.
        overall_wer: Mean Word Error Rate across all samples.
        overall_cer: Mean Character Error Rate across all samples.
        avg_time_sec: Mean processing duration in seconds.
        group_stats: Mapping of condition group (e.g. ``clean``, ``degraded``) to
            ``count``, ``mean_wer``, ``mean_cer``, and ``mean_time_sec``.
        rows: Per-sample display dictionaries (``ID``, ``Group``, ``WER``, ``CER``,
            ``Processing Time (s)``).
    """

    total_samples: int
    overall_wer: float
    overall_cer: float
    avg_time_sec: float
    group_stats: dict[str, dict[str, float | int]]
    rows: list[dict[str, Any]]


def compute_content_hash(file_bytes: bytes) -> str:
    """Compute a deterministic SHA-256 hex digest for uploaded file bytes.

    Args:
        file_bytes: Raw file payload.

    Returns:
        64-character lowercase SHA-256 hex string.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def extract_uploaded_bytes(
    filename: str,
    file_bytes: bytes,
    *,
    ocr_dpi: int = config.OCR_DPI,
    apply_threshold: bool = False,
) -> ExtractionResult:
    """Write uploaded bytes to an isolated temporary file, extract, and clean up.

    Preserves Unicode/Turkish filenames safely on Windows and guarantees the
    temporary file is deleted immediately on both success and failure.

    Args:
        filename: Uploaded filename (directory components are stripped).
        file_bytes: Raw file bytes.
        ocr_dpi: PDF rasterization DPI for OCR fallback.
        apply_threshold: Whether to enable adaptive thresholding during OCR.

    Returns:
        :class:`ExtractionResult` with ``filename`` and ``file_path`` set to the
        sanitized uploaded filename rather than a transient temp path.
    """
    safe_name = Path(filename).name or "uploaded_file"
    tmp_dir = Path(tempfile.mkdtemp())
    temp_path = tmp_dir / safe_name
    result: ExtractionResult | None = None
    caught_exc: Exception | None = None
    try:
        temp_path.write_bytes(file_bytes)
        result = extract_text(
            temp_path,
            ocr_dpi=ocr_dpi,
            apply_threshold=apply_threshold,
        )
        result.filename = safe_name
        result.file_path = safe_name
    except Exception as exc:
        # Construct a clean exception without retaining native PyMuPDF file handles
        # inside chained tracebacks so Windows can unlink the temp file immediately.
        caught_exc = type(exc)(str(exc))
    finally:
        gc.collect()
        shutil.rmtree(tmp_dir, ignore_errors=False)

    if caught_exc is not None:
        raise caught_exc
    if result is None:  # pragma: no cover - defensive guard
        raise CorruptedFileError(f"Extraction returned no result for '{safe_name}'.")
    return result


def add_items_to_workspace(
    workspace: list[WorkspaceItem],
    incoming_files: list[tuple[str, bytes]],
    *,
    ocr_dpi: int = config.OCR_DPI,
    apply_threshold: bool = False,
) -> tuple[list[WorkspaceItem], list[str], list[str], list[str]]:
    """Append incoming files to an existing workspace with SHA-256 deduplication.

    Valid files are extracted and appended without discarding existing workspace
    entries. Duplicate byte payloads are skipped gracefully, and per-file errors
    do not abort other valid files in the same batch.

    Args:
        workspace: Existing list of :class:`WorkspaceItem` objects.
        incoming_files: Sequence of ``(filename, file_bytes)`` tuples.
        ocr_dpi: PDF OCR DPI setting applied to newly added files.
        apply_threshold: Adaptive thresholding flag applied to newly added files.

    Returns:
        Tuple of ``(updated_workspace, added_names, duplicate_names, error_messages)``.
    """
    updated = list(workspace)
    seen_hashes = {item.content_hash for item in updated}
    added_names: list[str] = []
    duplicate_names: list[str] = []
    error_messages: list[str] = []

    for raw_name, file_bytes in incoming_files:
        safe_name = Path(raw_name).name or "uploaded_file"
        content_hash = compute_content_hash(file_bytes)
        if content_hash in seen_hashes:
            duplicate_names.append(safe_name)
            logger.info("Skipped duplicate upload '%s' (%s).", safe_name, content_hash[:8])
            continue

        try:
            result = extract_uploaded_bytes(
                safe_name,
                file_bytes,
                ocr_dpi=ocr_dpi,
                apply_threshold=apply_threshold,
            )
        except (ExtractionError, OSError) as exc:
            logger.warning("Failed to extract '%s': %s", safe_name, exc)
            error_messages.append(f"{safe_name}: {exc}")
            continue

        updated.append(
            WorkspaceItem(
                content_hash=content_hash,
                filename=safe_name,
                extension=Path(safe_name).suffix.lower(),
                file_bytes=file_bytes,
                size_bytes=len(file_bytes),
                result=result,
                ocr_dpi=ocr_dpi,
                apply_threshold=apply_threshold,
            )
        )
        seen_hashes.add(content_hash)
        added_names.append(safe_name)

    return updated, added_names, duplicate_names, error_messages


def remove_workspace_item(
    workspace: list[WorkspaceItem],
    content_hash: str,
) -> list[WorkspaceItem]:
    """Remove a single item matching ``content_hash`` from the workspace.

    Args:
        workspace: Current workspace items.
        content_hash: SHA-256 digest of the item to remove.

    Returns:
        New workspace list excluding the removed item.
    """
    return [item for item in workspace if item.content_hash != content_hash]


def reprocess_workspace(
    workspace: list[WorkspaceItem],
    *,
    ocr_dpi: int,
    apply_threshold: bool,
) -> tuple[list[WorkspaceItem], list[str]]:
    """Re-run extraction on all loaded workspace items with updated OCR settings.

    Args:
        workspace: Current workspace items.
        ocr_dpi: New PDF OCR rendering DPI.
        apply_threshold: New adaptive thresholding toggle.

    Returns:
        Tuple of ``(updated_workspace, error_messages)``. Items that fail
        reprocessing retain their previous extraction result.
    """
    updated: list[WorkspaceItem] = []
    errors: list[str] = []

    for item in workspace:
        try:
            new_result = extract_uploaded_bytes(
                item.filename,
                item.file_bytes,
                ocr_dpi=ocr_dpi,
                apply_threshold=apply_threshold,
            )
            updated.append(
                WorkspaceItem(
                    content_hash=item.content_hash,
                    filename=item.filename,
                    extension=item.extension,
                    file_bytes=item.file_bytes,
                    size_bytes=item.size_bytes,
                    result=new_result,
                    ocr_dpi=ocr_dpi,
                    apply_threshold=apply_threshold,
                )
            )
        except (ExtractionError, OSError) as exc:
            logger.warning("Reprocessing failed for '%s': %s", item.filename, exc)
            errors.append(f"{item.filename}: {exc}")
            updated.append(item)

    return updated, errors


def render_pdf_preview(
    pdf_bytes: bytes,
    page_index: int = 0,
    preview_dpi: int = 150,
) -> tuple[np.ndarray, int]:
    """Render a specific PDF page from bytes into an RGB numpy array.

    Args:
        pdf_bytes: Raw PDF document bytes.
        page_index: Zero-based page index (clamped safely to valid bounds).
        preview_dpi: Rendering resolution for visual preview.

    Returns:
        Tuple of ``(rgb_image_array, total_page_count)``.

    Raises:
        CorruptedFileError: If the PDF bytes cannot be opened or contain no pages.
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise CorruptedFileError(f"Unable to render PDF preview: {exc}") from exc

    with doc:
        page_count = doc.page_count
        if page_count == 0:
            raise CorruptedFileError("PDF document contains zero pages.")
        clamped_index = max(0, min(page_index, page_count - 1))
        pix = doc[clamped_index].get_pixmap(dpi=preview_dpi, alpha=False)
        rgb = (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, pix.n)
            .copy()
        )
        return rgb, page_count


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes in memory into an RGB numpy array.

    Args:
        image_bytes: Raw PNG/JPG/JPEG bytes.

    Returns:
        3-channel uint8 RGB image array.

    Raises:
        CorruptedFileError: If bytes are empty or unreadable by OpenCV.
    """
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    if buffer.size == 0:
        raise CorruptedFileError("Image payload is empty.")
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise CorruptedFileError("Unable to decode image bytes.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_preprocessed_preview(
    rgb_image: np.ndarray,
    *,
    apply_threshold: bool = False,
) -> np.ndarray:
    """Generate the exact OpenCV preprocessed view passed into Tesseract OCR.

    Args:
        rgb_image: Original RGB (or 2D grayscale) preview image array.
        apply_threshold: Whether to apply adaptive thresholding.

    Returns:
        2D uint8 grayscale (or binary) preprocessed image array.
    """
    bgr = (
        cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if rgb_image.ndim == 3
        else rgb_image
    )
    return preprocess_image(
        bgr,
        apply_deskew=True,
        apply_threshold=apply_threshold,
    )


def compute_cleaning_stats(raw_text: str, cleaned_text: str) -> CleaningStats:
    """Compute raw-vs-cleaned character reduction stats and preserved technical terms.

    Args:
        raw_text: Uncleaned extracted text.
        cleaned_text: Output from :func:`src.cleaner.clean_text`.

    Returns:
        Populated :class:`CleaningStats` instance.
    """
    raw_chars = len(raw_text)
    cleaned_chars = len(cleaned_text)
    removed_chars = max(0, raw_chars - cleaned_chars)
    reduction_pct = (
        round((removed_chars / raw_chars) * 100.0, 2) if raw_chars > 0 else 0.0
    )
    preserved = [
        label
        for label, pattern in _TECH_TERM_PATTERNS
        if pattern.search(cleaned_text) is not None
    ]
    return CleaningStats(
        raw_chars=raw_chars,
        cleaned_chars=cleaned_chars,
        removed_chars=removed_chars,
        reduction_pct=reduction_pct,
        preserved_tech_terms=preserved,
    )


def export_workspace_json(workspace: list[WorkspaceItem]) -> str:
    """Serialize all workspace extraction results into formatted UTF-8 JSON.

    Args:
        workspace: Workspace items to export.

    Returns:
        Pretty-printed JSON string containing all ``ExtractionResult`` dicts.
    """
    payload = [item.result.to_dict() for item in workspace]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_workspace_csv(workspace: list[WorkspaceItem]) -> str:
    """Serialize workspace summary metadata into a UTF-8 CSV string.

    Args:
        workspace: Workspace items to export.

    Returns:
        CSV text with headers ``Filename,Method,Pages,Cleaned Chars,Time (s),Warnings``.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        ["Filename", "Method", "Pages", "Cleaned Chars", "Time (s)", "Warnings"]
    )
    for item in workspace:
        res = item.result
        warnings_text = "; ".join(res.warnings) if res.warnings else "None"
        writer.writerow(
            [
                res.filename,
                res.method,
                res.page_count,
                len(res.cleaned_text),
                f"{res.processing_time_sec:.2f}",
                warnings_text,
            ]
        )
    return output.getvalue()


def find_sample_cv(dataset_dir: Path = config.DEFAULT_BENCHMARK_DIR) -> Path | None:
    """Locate a deterministic local sample PDF for the optional demo button.

    Args:
        dataset_dir: Local dataset directory to inspect (read-only).

    Returns:
        First sorted ``.pdf`` path inside ``dataset_dir``, or ``None`` if unavailable.
    """
    directory = Path(dataset_dir)
    if not directory.is_dir():
        return None
    pdfs = sorted(
        (p for p in directory.glob("*.pdf") if p.is_file()),
        key=lambda p: p.name,
    )
    return pdfs[0] if pdfs else None


def load_benchmark_summary(
    csv_path: Path = config.BENCHMARK_OUTPUT_PATH,
) -> BenchmarkSummary | None:
    """Load and aggregate ``outputs/ocr_benchmark.csv`` for the Benchmark UI tab.

    Supports both Phase 1 evaluator headers (``resume_id``, ``condition``) and
    normalized display headers (``ID``, ``group``).

    Args:
        csv_path: Path to the benchmark CSV file.

    Returns:
        :class:`BenchmarkSummary` when data is available, or ``None`` if missing/empty.
    """
    path = Path(csv_path)
    if not path.is_file():
        return None

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            resume_id = str(raw_row.get("resume_id") or raw_row.get("ID") or "").strip()
            group = str(
                raw_row.get("condition") or raw_row.get("group") or "unknown"
            ).strip()
            if not resume_id:
                continue
            try:
                wer = float(raw_row["wer"])
                cer = float(raw_row["cer"])
                proc_time = float(raw_row["processing_time_sec"])
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed benchmark row: %s", raw_row)
                continue

            entry = {
                "ID": resume_id,
                "Group": group,
                "WER": round(wer, 4),
                "CER": round(cer, 4),
                "Processing Time (s)": round(proc_time, 2),
            }
            rows.append(entry)
            grouped.setdefault(group, []).append(entry)

    if not rows:
        return None

    total = len(rows)
    overall_wer = sum(r["WER"] for r in rows) / total
    overall_cer = sum(r["CER"] for r in rows) / total
    avg_time = sum(r["Processing Time (s)"] for r in rows) / total

    group_stats: dict[str, dict[str, float | int]] = {}
    for group_name, group_rows in grouped.items():
        count = len(group_rows)
        group_stats[group_name] = {
            "count": count,
            "mean_wer": round(sum(r["WER"] for r in group_rows) / count, 4),
            "mean_cer": round(sum(r["CER"] for r in group_rows) / count, 4),
            "mean_time_sec": round(
                sum(r["Processing Time (s)"] for r in group_rows) / count, 2
            ),
        }

    return BenchmarkSummary(
        total_samples=total,
        overall_wer=round(overall_wer, 4),
        overall_cer=round(overall_cer, 4),
        avg_time_sec=round(avg_time, 2),
        group_stats=group_stats,
        rows=rows,
    )
