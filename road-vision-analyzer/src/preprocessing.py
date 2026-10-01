"""Reusable image preprocessing helpers shared by all detectors."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .utils import ImageLoadError, OutputError


def load_image(path: Path | str) -> np.ndarray:
    """Load a BGR image. Raises ImageLoadError for missing/empty/corrupted files."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Image file not found: {path}")
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII paths
    except OSError as exc:
        raise ImageLoadError(f"Cannot read image '{path}': {exc}") from exc
    if raw.size == 0:
        raise ImageLoadError(f"Image file is empty: {path}")
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Invalid or corrupted image (cannot decode): {path}")
    return image


def save_image(path: Path | str, image: np.ndarray) -> None:
    """Save an image; the format is chosen from the file extension."""
    path = Path(path)
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OutputError(f"Cannot encode image for saving: {path}")
    try:
        buffer.tofile(str(path))
    except OSError as exc:
        raise OutputError(f"Cannot write image '{path}': {exc}") from exc


def resize_to_max_dimension(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Down-scale so the longest side is <= max_dim. Returns (image, scale_factor)."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / longest
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR -> grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """BGR -> HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian blur with an odd square kernel."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def otsu_threshold(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    """Binary threshold with Otsu's automatic level."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return binary


def morph_close(mask: np.ndarray, kernel_size: tuple[int, int] | int = 5, iterations: int = 1) -> np.ndarray:
    """Morphological closing (fills small gaps/holes)."""
    size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morph_open(mask: np.ndarray, kernel_size: tuple[int, int] | int = 3, iterations: int = 1) -> np.ndarray:
    """Morphological opening (removes small noise)."""
    size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def detect_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detection."""
    return cv2.Canny(gray, low, high)
