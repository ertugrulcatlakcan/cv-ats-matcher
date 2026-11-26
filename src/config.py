"""Centralized configuration constants and paths for Phase 1 ATS ingestion."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Project and dataset paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
RESUME_CSV_PATH: Path = DATASET_DIR / "Resume.csv"
PDF_DATA_DIR: Path = DATASET_DIR / "data"
DEFAULT_BENCHMARK_DIR: Path = PDF_DATA_DIR / "INFORMATION-TECHNOLOGY"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
BENCHMARK_OUTPUT_PATH: Path = OUTPUTS_DIR / "ocr_benchmark.csv"

# Smart-router and OCR parameters
MIN_TEXT_LENGTH: int = 50
OCR_DPI: int = 300
TESSERACT_LANG: str = "eng"
TESSERACT_CONFIG: str = "--oem 3 --psm 3"
RANDOM_SEED: int = 42

# Deskew bounds (degrees)
DESKEW_MIN_ANGLE: float = 0.5
DESKEW_MAX_ANGLE: float = 15.0

# Supported input extensions (lowercase)
SUPPORTED_PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
SUPPORTED_DOCX_EXTENSIONS: frozenset[str] = frozenset({".docx"})
SUPPORTED_EXTENSIONS: frozenset[str] = (
    SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_DOCX_EXTENSIONS
)

_WINDOWS_TESSERACT_FALLBACKS: tuple[Path, ...] = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


def resolve_tesseract_cmd(explicit_cmd: str | None = None) -> str | None:
    """Resolve the Tesseract OCR executable path.

    Checks in priority order: explicit argument, ``TESSERACT_CMD`` environment
    variable, system ``PATH``, and common Windows installation directories.

    Args:
        explicit_cmd: Optional explicit executable path override.

    Returns:
        Resolved executable path as a string, or ``None`` if unavailable.
    """
    if explicit_cmd:
        explicit_path = Path(explicit_cmd)
        if explicit_path.is_file():
            return str(explicit_path)
        found = shutil.which(explicit_cmd)
        if found:
            return found

    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        env_path = Path(env_cmd)
        if env_path.is_file():
            return str(env_path)
        found_env = shutil.which(env_cmd)
        if found_env:
            return found_env

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd

    for fallback in _WINDOWS_TESSERACT_FALLBACKS:
        if fallback.is_file():
            logger.debug("Resolved Tesseract from fallback path: %s", fallback)
            return str(fallback)

    return None


TESSERACT_CMD: str | None = resolve_tesseract_cmd()
