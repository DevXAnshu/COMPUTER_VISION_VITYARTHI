"""Rule-based traffic sign detection and classification (classical Computer Vision).

Pipeline: BGR -> HSV -> colour mask (red/blue/yellow) -> morphology -> contours ->
geometric filtering -> shape analysis -> rule-based classification -> heuristic confidence.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from . import preprocessing as prep
from .config import (DARK_MAX_VALUE, HSV_RANGES, SIGN_PARAMS, WHITE_MAX_SATURATION,
                     WHITE_MIN_VALUE, SignParams)
from .utils import make_bbox, suppress_overlaps

SIGN_TYPES = ("STOP", "SPEED_LIMIT", "NO_ENTRY", "WARNING", "DIRECTION", "UNKNOWN")

# Shapes accepted as a good match for each sign type (used by the confidence score).
EXPECTED_SHAPES = {
    "STOP": {"Octagon"},
    "SPEED_LIMIT": {"Circle"},
    "NO_ENTRY": {"Circle"},
    "WARNING": {"Triangle", "Square"},
    "DIRECTION": {"Square", "Rectangle", "Circle"},
}


# ----------------------------------------------------------------------------- colour masks
def build_color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """Binary mask of pixels whose HSV value falls in the configured range(s) of a colour."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    return mask


def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Close small holes (e.g. letters inside a sign) then remove small noise."""
    return prep.morph_open(prep.morph_close(mask, kernel_size), kernel_size)


# ----------------------------------------------------------------------------- candidates
def find_sign_candidates(image: np.ndarray, params: SignParams = SIGN_PARAMS) -> list[dict[str, Any]]:
    """Locate coloured regions that could be traffic signs (no classification yet)."""
    height, width = image.shape[:2]
    image_area = float(height * width)
    hsv = prep.to_hsv(prep.gaussian_blur(image, params.blur_ksize))
    candidates: list[dict[str, Any]] = []

    for color in HSV_RANGES:
        mask = clean_mask(build_color_mask(hsv, color), params.morph_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (params.min_area_ratio * image_area <= area <= params.max_area_ratio * image_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < params.min_side_px:
                continue
            if not (params.min_aspect <= w / h <= params.max_aspect):
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0 or area / hull_area < params.min_solidity:
                continue
            candidates.append({
                "color": color, "hull": hull, "bbox": (x, y, w, h),
                "solidity": area / hull_area, "mask": mask, "hsv": hsv,
            })
    return candidates


# ----------------------------------------------------------------------------- shape analysis
def analyze_shape(hull: np.ndarray, bbox: tuple[int, int, int, int],
                  params: SignParams = SIGN_PARAMS) -> dict[str, Any]:
    """Describe a contour: polygon sides, circularity and a shape name."""
    area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, params.approx_epsilon * perimeter, True)
    sides = len(approx)
    circularity = 4.0 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    (_, _), radius = cv2.minEnclosingCircle(hull)
    # Area ratio vs. the smallest enclosing circle: ~1.0 circle, ~0.90 octagon, 0.64 square, 0.41 triangle.
    circle_ratio = area / (math.pi * radius ** 2) if radius > 0 else 0.0
    _, _, w, h = bbox
    aspect = w / h

    if circularity >= params.circle_circularity and circle_ratio >= params.circle_ratio:
        shape = "Circle"
    elif sides == 3:
        shape = "Triangle"
    elif sides == 4:
        shape = "Square" if 0.85 <= aspect <= 1.18 else "Rectangle"
    elif 7 <= sides <= 9:
        shape = "Octagon"
    else:
        shape = "Polygon"
    return {"shape": shape, "sides": sides, "circularity": circularity,
            "circle_ratio": circle_ratio, "aspect": aspect}


# ----------------------------------------------------------------------------- interior features
def extract_interior_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Measure colour coverage and white/dark patterns inside the candidate shape."""
    x, y, w, h = candidate["bbox"]
    shifted_hull = candidate["hull"] - np.array([[x, y]])
    shape_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(shape_mask, [shifted_hull.astype(np.int32)], -1, 255, thickness=-1)
    shape_area = max(1, int(np.count_nonzero(shape_mask)))

    roi_hsv = candidate["hsv"][y:y + h, x:x + w]
    roi_color = candidate["mask"][y:y + h, x:x + w]
    inside = shape_mask > 0
    white = (roi_hsv[..., 1] < WHITE_MAX_SATURATION) & (roi_hsv[..., 2] > WHITE_MIN_VALUE) & inside
    dark = (roi_hsv[..., 2] < DARK_MAX_VALUE) & inside

    # Central window (60% x 60%) and central horizontal strip (60% wide x 20% tall).
    cy0, cy1, cx0, cx1 = int(0.2 * h), int(0.8 * h), int(0.2 * w), int(0.8 * w)
    by0, by1 = int(0.4 * h), int(0.6 * h)
    center_dark = dark[cy0:cy1, cx0:cx1]
    bar_white = white[by0:max(by1, by0 + 1), cx0:cx1]
    return {
        "color_fraction": float(np.count_nonzero((roi_color > 0) & inside)) / shape_area,
        "white_ratio": float(np.count_nonzero(white)) / shape_area,
        "center_dark_ratio": float(np.count_nonzero(center_dark)) / max(1, center_dark.size),
        "bar_white_ratio": float(np.count_nonzero(bar_white)) / max(1, bar_white.size),
    }


# ----------------------------------------------------------------------------- classification
def classify_sign(color: str, shape: str, features: dict[str, float]) -> tuple[str, float]:
    """Rule-based classification. Returns (sign_type, evidence_score in 0..1)."""
    if color == "red":
        if shape == "Octagon":
            return "STOP", 1.0 if features["white_ratio"] > 0.03 else 0.7
        if shape == "Circle":
            if features["color_fraction"] >= 0.6:  # mostly solid red disc
                if features["bar_white_ratio"] > 0.6:
                    return "NO_ENTRY", 1.0
            elif features["center_dark_ratio"] > 0.02:  # red ring + dark digits inside
                return "SPEED_LIMIT", 1.0
        if shape == "Triangle":
            return "WARNING", 1.0 if features["white_ratio"] > 0.2 else 0.7
    elif color == "yellow":
        if shape in ("Triangle", "Square"):
            return "WARNING", 0.8
    elif color == "blue":
        if shape in ("Square", "Rectangle", "Circle"):
            return "DIRECTION", 1.0 if features["white_ratio"] > 0.03 else 0.6
    return "UNKNOWN", 0.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_confidence(sign_type: str, shape: str, evidence: float, solidity: float,
                       features: dict[str, float], bbox: tuple[int, int, int, int],
                       image_area: float) -> float:
    """Heuristic confidence (0-100): weighted rule evidence, NOT a machine-learning probability."""
    x, y, w, h = bbox
    contour_quality = _clip01((solidity - 0.8) / 0.2)
    shape_match = 1.0 if shape in EXPECTED_SHAPES.get(sign_type, set()) else 0.3
    color_score = _clip01(features["color_fraction"] / 0.4)
    aspect_score = _clip01(1.0 - abs(math.log(w / h)))
    size_score = 0.5 * aspect_score + 0.5 * min(1.0, (w * h) / (0.004 * image_area))
    score = (0.20 * contour_quality + 0.20 * shape_match + 0.20 * color_score
             + 0.10 * size_score + 0.30 * evidence)
    confidence = 100.0 * score
    if sign_type == "UNKNOWN":
        confidence = min(50.0, confidence * 0.5)
    return round(confidence, 1)


# ----------------------------------------------------------------------------- public API
def detect_traffic_signs(image: np.ndarray, params: SignParams = SIGN_PARAMS) -> list[dict[str, Any]]:
    """Detect and classify traffic signs. Returns a list of result dictionaries."""
    height, width = image.shape[:2]
    image_area = float(height * width)
    results: list[dict[str, Any]] = []

    for cand in find_sign_candidates(image, params):
        shape_info = analyze_shape(cand["hull"], cand["bbox"], params)
        features = extract_interior_features(cand)
        sign_type, evidence = classify_sign(cand["color"], shape_info["shape"], features)
        confidence = compute_confidence(sign_type, shape_info["shape"], evidence,
                                        cand["solidity"], features, cand["bbox"], image_area)
        if confidence < params.min_confidence:
            continue
        x, y, w, h = cand["bbox"]
        results.append({
            "type": sign_type,
            "shape": shape_info["shape"],
            "color": cand["color"].capitalize(),
            "confidence": confidence,
            "bounding_box": make_bbox(x, y, w, h),
            "polygon_sides": shape_info["sides"],
        })
    return sorted(suppress_overlaps(results, params.overlap_iou), key=lambda r: r["bounding_box"]["x"])
