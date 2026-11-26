"""Unit tests for src.preprocessor."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.models import CorruptedFileError
from src.preprocessor import (
    deskew_image,
    load_image_unicode_safe,
    preprocess_image,
)


def _make_text_page(angle: float = 0.0) -> np.ndarray:
    """Generate a synthetic high-contrast document page image."""
    img = Image.new("RGB", (800, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(60, 340, 40):
        draw.rectangle([80, y, 720, y + 12], fill=(0, 0, 0))
    arr = np.array(img)
    if angle != 0.0:
        height, width = arr.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        arr = cv2.warpAffine(arr, matrix, (width, height), borderValue=(255, 255, 255))
    return arr


def test_load_image_unicode_safe_with_turkish_path(tmp_path: Path) -> None:
    """Images stored under Turkish Unicode Windows paths load cleanly."""
    unicode_dir = tmp_path / "özgeçmiş_şğüöçı_İ"
    unicode_dir.mkdir()
    img_path = unicode_dir / "cv_örnek.png"
    Image.fromarray(_make_text_page()).save(img_path)

    loaded = load_image_unicode_safe(img_path)
    assert loaded.shape[:2] == (400, 800)


def test_load_image_unicode_safe_raises_on_corrupted(tmp_path: Path) -> None:
    """Unreadable image bytes raise CorruptedFileError."""
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not-a-valid-image")

    with pytest.raises(CorruptedFileError):
        load_image_unicode_safe(bad)


def test_load_image_unicode_safe_raises_on_missing(tmp_path: Path) -> None:
    """Missing image path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_image_unicode_safe(tmp_path / "missing.png")


def test_deskew_skips_straight_image() -> None:
    """Straight pages are not unnecessarily rotated."""
    gray = cv2.cvtColor(_make_text_page(0.0), cv2.COLOR_RGB2GRAY)
    _, applied_angle = deskew_image(gray)
    assert applied_angle == 0.0


def test_deskew_corrects_tilted_image_within_bounds() -> None:
    """Tilted pages within 0.5-15 degrees are deskewed back toward horizontal."""
    gray = cv2.cvtColor(_make_text_page(3.0), cv2.COLOR_RGB2GRAY)
    _, applied_angle = deskew_image(gray)
    assert 1.5 <= abs(applied_angle) <= 4.5


def test_preprocess_threshold_off_by_default_and_optional() -> None:
    """Adaptive thresholding defaults to off and outputs binary values when enabled."""
    rgb = _make_text_page(0.0)
    gray_out = preprocess_image(rgb, apply_deskew=False, apply_threshold=False)
    bin_out = preprocess_image(rgb, apply_deskew=False, apply_threshold=True)

    assert gray_out.ndim == 2
    assert bin_out.ndim == 2
    assert set(np.unique(bin_out)).issubset({0, 255})
