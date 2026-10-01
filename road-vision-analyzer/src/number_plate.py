"""Number plate detection using classical Computer Vision.

Pipeline: grayscale -> Gaussian blur -> Canny edges -> morphological closing -> contours ->
size / aspect-ratio / rectangularity filtering -> heuristic ranking -> overlap suppression -> crop.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import preprocessing as prep
from .config import PLATE_PARAMS, PlateParams
from .utils import make_bbox, suppress_overlaps


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _range_score(value: float, ideal_low: float, ideal_high: float, falloff: float) -> float:
    """1.0 inside [ideal_low, ideal_high]; decreases linearly to 0 over `falloff` outside."""
    if ideal_low <= value <= ideal_high:
        return 1.0
    distance = ideal_low - value if value < ideal_low else value - ideal_high
    return _clip01(1.0 - distance / falloff)


def compute_plate_confidence(aspect: float, area_ratio: float, rectangularity: float,
                             edge_density: float, center_y_ratio: float) -> float:
    """Heuristic confidence (0-100) from five geometric cues. Not a learned probability."""
    aspect_score = _range_score(aspect, 2.5, 5.5, 2.0)
    area_score = _range_score(area_ratio, 0.004, 0.08, 0.04)
    rect_score = _clip01((rectangularity - 0.45) / 0.45)
    edge_score = _range_score(edge_density, 0.06, 0.35, 0.15)
    position_score = 1.0 if 0.35 <= center_y_ratio <= 0.95 else 0.6
    score = (0.25 * aspect_score + 0.15 * area_score + 0.25 * rect_score
             + 0.25 * edge_score + 0.10 * position_score)
    return round(100.0 * score, 1)


def find_plate_candidates(image: np.ndarray, params: PlateParams = PLATE_PARAMS) -> list[dict[str, Any]]:
    """Return scored plate-like rectangles (before overlap suppression and thresholding)."""
    height, width = image.shape[:2]
    gray = prep.gaussian_blur(prep.to_gray(image), params.blur_ksize)
    edges = prep.detect_edges(gray, params.canny_low, params.canny_high)

    # Wide, short kernel joins the characters and border of a plate into one blob.
    kernel_w = max(9, (width // 60) | 1)
    closed = prep.morph_close(edges, (kernel_w, 5))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < params.min_height_px or h == 0:
            continue
        if not (params.min_width_ratio * width <= w <= params.max_width_ratio * width):
            continue
        aspect = w / h
        if not (params.min_aspect <= aspect <= params.max_aspect):
            continue
        rectangularity = cv2.contourArea(contour) / float(w * h)
        if rectangularity < params.min_rectangularity:
            continue
        edge_density = float(np.count_nonzero(edges[y:y + h, x:x + w])) / (w * h)
        area_ratio = (w * h) / float(width * height)
        confidence = compute_plate_confidence(aspect, area_ratio, rectangularity,
                                              edge_density, (y + h / 2) / height)
        candidates.append({
            "detected": True,
            "confidence": confidence,
            "bounding_box": make_bbox(x, y, w, h),
            "area": int(w * h),
            "aspect_ratio": round(aspect, 2),
        })
    return candidates


def detect_number_plates(image: np.ndarray, params: PlateParams = PLATE_PARAMS) -> list[dict[str, Any]]:
    """Detect likely number plates; returns up to `max_plates` results ranked by confidence."""
    candidates = [c for c in find_plate_candidates(image, params)
                  if c["confidence"] >= params.min_confidence]
    ranked = suppress_overlaps(candidates, params.overlap_iou)
    return ranked[:params.max_plates]


def crop_plate(image: np.ndarray, plate: dict[str, Any], padding: int = PLATE_PARAMS.crop_padding) -> np.ndarray:
    """Crop the plate region (with a small padding) from the image."""
    box = plate["bounding_box"]
    height, width = image.shape[:2]
    x0, y0 = max(0, box["x"] - padding), max(0, box["y"] - padding)
    x1 = min(width, box["x"] + box["width"] + padding)
    y1 = min(height, box["y"] + box["height"] + padding)
    return image[y0:y1, x0:x1].copy()
