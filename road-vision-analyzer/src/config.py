"""Centralised configuration: HSV colour ranges and tunable detection parameters.

All thresholds live here so they can be tuned without touching algorithm code.
HSV note: OpenCV uses H in [0, 179], S and V in [0, 255].
"""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Images larger than this (longest side, in pixels) are down-scaled before analysis.
# Reported coordinates refer to the processed (possibly down-scaled) image.
MAX_PROCESS_DIMENSION = 1280

# Colour name -> list of (lower HSV, upper HSV). Red wraps around hue 0, so it has two ranges.
HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (179, 255, 255))],
    "blue": [((100, 120, 50), (130, 255, 255))],
    "yellow": [((18, 110, 110), (35, 255, 255))],
}

# "White" and "dark" definitions used for analysing the inside of a sign.
WHITE_MAX_SATURATION = 70
WHITE_MIN_VALUE = 170
DARK_MAX_VALUE = 100


@dataclass(frozen=True)
class SignParams:
    """Tunable parameters for traffic sign detection."""

    blur_ksize: int = 5
    morph_kernel: int = 5
    min_area_ratio: float = 0.0005   # min contour area as a fraction of the image area
    max_area_ratio: float = 0.5
    min_side_px: int = 20
    min_aspect: float = 0.5          # bounding-box width / height
    max_aspect: float = 2.5
    min_solidity: float = 0.85       # contour area / convex hull area
    approx_epsilon: float = 0.02     # approxPolyDP epsilon as a fraction of the perimeter
    circle_circularity: float = 0.85
    circle_ratio: float = 0.93       # contour area / min enclosing circle area
    min_confidence: float = 35.0
    overlap_iou: float = 0.4


@dataclass(frozen=True)
class PlateParams:
    """Tunable parameters for number plate detection."""

    blur_ksize: int = 5
    canny_low: int = 50
    canny_high: int = 150
    min_width_ratio: float = 0.07    # plate width as a fraction of image width
    max_width_ratio: float = 0.5
    min_height_px: int = 12
    min_aspect: float = 2.0          # width / height
    max_aspect: float = 6.5
    min_rectangularity: float = 0.45  # contour area / bounding-box area
    min_confidence: float = 50.0
    max_plates: int = 3
    overlap_iou: float = 0.3
    crop_padding: int = 4


SIGN_PARAMS = SignParams()
PLATE_PARAMS = PlateParams()
