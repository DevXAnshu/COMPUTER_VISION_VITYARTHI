"""Result combination and annotated-image drawing for both detectors."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .utils import bbox_tuple, box_containment

SIGN_COLOR = (0, 200, 0)      # BGR green
PLATE_COLOR = (0, 140, 255)   # BGR orange


def remove_plates_inside_signs(plates: list[dict], signs: list[dict],
                               max_containment: float = 0.6) -> list[dict]:
    """Drop plate candidates that lie mostly inside a detected sign (e.g. the text or bar on a sign)."""
    return [p for p in plates
            if all(box_containment(bbox_tuple(p["bounding_box"]), bbox_tuple(s["bounding_box"])) <= max_containment
                   for s in signs)]


def combine_results(image_name: str, image_size: tuple[int, int], signs: list[dict],
                    plates: list[dict]) -> dict[str, Any]:
    """Merge sign and plate detections into one per-image result dictionary."""
    return {
        "image": image_name,
        "status": "ok",
        "error": None,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "traffic_signs": signs,
        "number_plates": plates,
    }


def _draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int],
                scale: float, thickness: int) -> None:
    """Draw a text label on a filled background, above (x, y) or below if there is no room."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = y - th - baseline - 4
    if top < 0:
        top = y + 4
    cv2.rectangle(image, (x, top), (x + tw + 6, top + th + baseline + 4), color, -1)
    cv2.putText(image, text, (x + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


def draw_annotations(image: np.ndarray, signs: list[dict], plates: list[dict]) -> np.ndarray:
    """Return a copy of the image with boxes and labels drawn for all detections."""
    output = image.copy()
    scale = max(0.45, output.shape[1] / 1600.0)
    thickness = max(1, int(round(scale * 2)))

    for sign in signs:
        b = sign["bounding_box"]
        cv2.rectangle(output, (b["x"], b["y"]), (b["x"] + b["width"], b["y"] + b["height"]),
                      SIGN_COLOR, thickness + 1)
        _draw_label(output, f"{sign['type']} - {sign['confidence']:.0f}%",
                    b["x"], b["y"], SIGN_COLOR, scale, thickness)

    for plate in plates:
        b = plate["bounding_box"]
        cv2.rectangle(output, (b["x"], b["y"]), (b["x"] + b["width"], b["y"] + b["height"]),
                      PLATE_COLOR, thickness + 1)
        label = f"NUMBER PLATE - {plate['confidence']:.0f}%"
        if plate.get("ocr_text"):
            label += f" | {plate['ocr_text']}"
        _draw_label(output, label, b["x"], b["y"], PLATE_COLOR, scale, thickness)
    return output
