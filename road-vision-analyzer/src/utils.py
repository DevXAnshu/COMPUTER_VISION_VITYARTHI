"""Shared helpers: exceptions, file discovery, geometry, JSON output, terminal formatting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import SUPPORTED_EXTENSIONS

LINE_WIDTH = 45


# ----------------------------------------------------------------------------- exceptions
class AnalyzerError(Exception):
    """Base class for all expected, user-facing errors."""


class InputNotFoundError(AnalyzerError):
    """The input path does not exist."""


class ImageLoadError(AnalyzerError):
    """An image is missing, empty, corrupted or not decodable."""


class NoImagesFoundError(AnalyzerError):
    """A folder contained no supported images."""


class OutputError(AnalyzerError):
    """Output directory or file could not be created/written."""


# ----------------------------------------------------------------------------- files
def is_supported_image(path: Path) -> bool:
    """Return True if the file extension is a supported image type."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_images(input_path: Path) -> list[Path]:
    """Return the list of images to process for a file or a folder input."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise InputNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if not is_supported_image(input_path):
            raise AnalyzerError(
                f"Unsupported file type '{input_path.suffix}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        return [input_path]
    images = sorted(p for p in input_path.iterdir() if p.is_file() and is_supported_image(p))
    if not images:
        raise NoImagesFoundError(
            f"No supported images ({', '.join(SUPPORTED_EXTENSIONS)}) found in folder: {input_path}"
        )
    return images


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) or raise OutputError."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"Cannot create output directory '{path}': {exc}") from exc
    return Path(path)


def save_json(data: Any, path: Path) -> None:
    """Write data as pretty-printed UTF-8 JSON."""
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4, ensure_ascii=False)
    except OSError as exc:
        raise OutputError(f"Cannot write JSON report '{path}': {exc}") from exc


# ----------------------------------------------------------------------------- geometry
def make_bbox(x: int, y: int, w: int, h: int) -> dict[str, int]:
    """Build the bounding-box dictionary used in results."""
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}


def bbox_tuple(bbox: dict[str, int]) -> tuple[int, int, int, int]:
    """Convert a bounding-box dictionary to an (x, y, w, h) tuple."""
    return bbox["x"], bbox["y"], bbox["width"], bbox["height"]


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return float(ix * iy)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    inter = _intersection(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def box_containment(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection area divided by the smaller box area (detects nested boxes)."""
    smaller = min(a[2] * a[3], b[2] * b[3])
    return _intersection(a, b) / smaller if smaller > 0 else 0.0


def suppress_overlaps(items: list[dict], iou_threshold: float,
                      containment_threshold: float = 0.85) -> list[dict]:
    """Non-maximum suppression: keep the highest-confidence item among overlapping ones."""
    kept: list[dict] = []
    for item in sorted(items, key=lambda i: i["confidence"], reverse=True):
        box = bbox_tuple(item["bounding_box"])
        duplicate = any(
            box_iou(box, bbox_tuple(k["bounding_box"])) > iou_threshold
            or box_containment(box, bbox_tuple(k["bounding_box"])) > containment_threshold
            for k in kept
        )
        if not duplicate:
            kept.append(item)
    return kept


# ----------------------------------------------------------------------------- terminal output
_OCR_LABELS = {
    "not_requested": "Not requested",
    "unavailable": "Unavailable (pytesseract/Tesseract not installed)",
    "no_text": "No text recognised",
    "error": "OCR error",
}


def banner(title: str = "ROAD VISION ANALYZER") -> str:
    """Return the application banner."""
    bar = "=" * LINE_WIDTH
    return f"{bar}\n{title.center(LINE_WIDTH)}\n{bar}"


def _location(bbox: dict[str, int]) -> str:
    return f"({bbox['x']}, {bbox['y']}, {bbox['width']}, {bbox['height']})"


def format_image_report(result: dict) -> str:
    """Format one image result as the terminal report block."""
    rule = "-" * LINE_WIDTH
    lines = [f"Input: {result['image']}", ""]
    signs, plates = result["traffic_signs"], result["number_plates"]

    lines += ["TRAFFIC SIGNS", rule, f"Detected: {len(signs)}"]
    for i, sign in enumerate(signs, 1):
        lines += [
            "", f"[{i}]",
            f"Type       : {sign['type']}",
            f"Shape      : {sign['shape']}",
            f"Color      : {sign['color']}",
            f"Confidence : {sign['confidence']:.0f}%",
            f"Location   : {_location(sign['bounding_box'])}",
        ]
    if not signs:
        lines.append("(no traffic signs found)")

    lines += ["", "NUMBER PLATES", rule, f"Detected: {len(plates)}"]
    for i, plate in enumerate(plates, 1):
        status = plate["ocr_status"]
        ocr_text = plate["ocr_text"] if status == "ok" else _OCR_LABELS.get(status, status)
        lines += [
            "", f"[{i}]",
            f"Confidence   : {plate['confidence']:.0f}%",
            f"Location     : {_location(plate['bounding_box'])}",
            f"Aspect Ratio : {plate['aspect_ratio']:.2f}",
            f"OCR          : {ocr_text}",
        ]
    if not plates:
        lines.append("(no number plates found)")

    lines += ["", rule]
    if result.get("annotated_image"):
        lines += ["Annotated image saved:", str(result["annotated_image"])]
    return "\n".join(lines)


def format_batch_summary(summary: dict) -> str:
    """Format the batch summary block."""
    rule = "-" * LINE_WIDTH
    lines = [
        "BATCH SUMMARY", rule,
        f"Images found     : {summary['total_images']}",
        f"Images processed : {summary['processed']}",
        f"Images failed    : {summary['failed']}",
        f"Traffic signs    : {summary['traffic_signs']}",
        f"Number plates    : {summary['number_plates']}",
    ]
    for name, reason in summary.get("failed_images", {}).items():
        lines.append(f"  failed: {name} -> {reason}")
    return "\n".join(lines)
