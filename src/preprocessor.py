"""Controlled OpenCV image preprocessing utilities for Tesseract 5 OCR."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import DESKEW_MAX_ANGLE, DESKEW_MIN_ANGLE
from src.models import CorruptedFileError

logger = logging.getLogger(__name__)

_PROJECTION_MAX_DIM = 800
_MIN_FOREGROUND_PIXELS = 20
_MIN_VARIANCE_IMPROVEMENT = 1.02


def load_image_unicode_safe(image_path: Path | str) -> np.ndarray:
    """Load an image file from disk safely on Windows paths with Unicode characters.

    Uses ``numpy.fromfile`` and ``cv2.imdecode`` instead of ``cv2.imread`` so
    paths containing Turkish or other non-ASCII characters never fail silently.

    Args:
        image_path: Path to the image file.

    Returns:
        Decoded BGR image array.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        CorruptedFileError: If the file is empty or cannot be decoded as an image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    if raw_bytes.size == 0:
        raise CorruptedFileError(f"Image file is empty: {path}")

    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise CorruptedFileError(f"Unable to decode image file: {path}")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image array to single-channel grayscale.

    Args:
        image: Input 2D grayscale, 3-channel BGR/RGB, or 4-channel BGRA/RGBA array.

    Returns:
        2D uint8 grayscale image array.
    """
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply mild Gaussian denoising suitable for Tesseract 5 LSTM recognition.

    Args:
        gray: 2D grayscale image array.
        kernel_size: Odd Gaussian kernel size (defaults to 3).

    Returns:
        Lightly denoised grayscale image array.
    """
    if kernel_size <= 1:
        return gray.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def estimate_skew_angle(
    gray: np.ndarray,
    max_angle: float = DESKEW_MAX_ANGLE,
) -> float:
    """Estimate the deskew rotation angle (in degrees) using projection variance.

    Tests candidate rotation corrections in ``[-max_angle, +max_angle]`` with
    0.5-degree increments and selects the angle that maximizes horizontal
    row-projection variance (sharply aligned text lines).

    Args:
        gray: 2D grayscale image array.
        max_angle: Maximum absolute tilt angle to search in degrees.

    Returns:
        Recommended counter-clockwise correction angle in degrees (``0.0`` when
        the page is already straight or blank).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if int(np.count_nonzero(binary)) < _MIN_FOREGROUND_PIXELS:
        return 0.0

    height, width = binary.shape[:2]
    max_dim = max(height, width)
    if max_dim > _PROJECTION_MAX_DIM:
        scale = _PROJECTION_MAX_DIM / float(max_dim)
        binary = cv2.resize(
            binary,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    b_height, b_width = binary.shape[:2]
    center = (b_width / 2.0, b_height / 2.0)

    def _projection_score(angle: float) -> float:
        if angle == 0.0:
            rotated = binary
        else:
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                binary,
                matrix,
                (b_width, b_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        row_sums = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(row_sums))

    base_score = _projection_score(0.0)
    best_angle = 0.0
    best_score = base_score

    step = 0.5
    steps = int(round(max_angle / step))
    for idx in range(-steps, steps + 1):
        candidate = idx * step
        if candidate == 0.0:
            continue
        score = _projection_score(candidate)
        if score > best_score:
            best_score = score
            best_angle = candidate

    if base_score > 0.0 and best_score < base_score * _MIN_VARIANCE_IMPROVEMENT:
        return 0.0

    return best_angle


def deskew_image(
    gray: np.ndarray,
    min_angle: float = DESKEW_MIN_ANGLE,
    max_angle: float = DESKEW_MAX_ANGLE,
) -> tuple[np.ndarray, float]:
    """Rotate a grayscale image only when detected skew is within meaningful bounds.

    Args:
        gray: 2D grayscale image array.
        min_angle: Minimum absolute angle (degrees) required to trigger rotation.
        max_angle: Maximum absolute angle (degrees) allowed for rotation.

    Returns:
        Tuple of ``(processed_image, applied_angle_degrees)``. Returns the original
        image and ``0.0`` when skew is outside ``[min_angle, max_angle]``.
    """
    correction_angle = estimate_skew_angle(gray, max_angle=max_angle)
    if not (min_angle <= abs(correction_angle) <= max_angle):
        logger.debug(
            "Skipping deskew (estimated angle %.2f deg outside [%.2f, %.2f]).",
            correction_angle,
            min_angle,
            max_angle,
        )
        return gray, 0.0

    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), correction_angle, 1.0)
    deskewed = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    logger.debug("Applied deskew correction of %.2f degrees.", correction_angle)
    return deskewed, correction_angle


def apply_adaptive_threshold(
    gray: np.ndarray,
    block_size: int = 31,
    c: int = 15,
) -> np.ndarray:
    """Apply Gaussian adaptive thresholding for uneven or shadowed scans.

    Args:
        gray: 2D grayscale image array.
        block_size: Odd pixel neighborhood size (must be >= 3).
        c: Constant subtracted from the local weighted mean.

    Returns:
        Binary uint8 image containing values 0 and 255.
    """
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def preprocess_image(
    image: np.ndarray,
    *,
    apply_deskew: bool = True,
    apply_threshold: bool = False,
) -> np.ndarray:
    """Run controlled preprocessing pipeline optimized for Tesseract 5 LSTM.

    Converts to grayscale, applies light Gaussian denoising, conditionally deskews
    when tilt lies in ``[DESKEW_MIN_ANGLE, DESKEW_MAX_ANGLE]``, and optionally
    applies adaptive thresholding (disabled by default).

    Args:
        image: Input image array (BGR, RGB, BGRA, or grayscale).
        apply_deskew: Whether to run conditional skew correction.
        apply_threshold: Whether to apply adaptive binarization (default ``False``).

    Returns:
        Preprocessed 2D uint8 image array ready for OCR.
    """
    processed = to_grayscale(image)
    processed = denoise_image(processed)
    if apply_deskew:
        processed, _ = deskew_image(processed)
    if apply_threshold:
        processed = apply_adaptive_threshold(processed)
    return processed
