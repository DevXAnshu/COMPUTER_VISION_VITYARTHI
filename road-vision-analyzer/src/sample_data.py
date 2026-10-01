"""Synthetic DEMO/TEST image generation.

These images are drawn programmatically with OpenCV. They are NOT real road photographs and
only exist so the project can be demonstrated and tested without external datasets.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import preprocessing as prep
from .utils import ensure_dir

RED, WHITE, BLUE, YELLOW, BLACK = (30, 30, 200), (255, 255, 255), (200, 100, 20), (0, 200, 240), (0, 0, 0)


def make_background(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """Grey vertical gradient with mild Gaussian noise (deterministic)."""
    rng = np.random.default_rng(seed)
    column = np.linspace(170, 110, height).reshape(-1, 1, 1)
    image = np.tile(column, (1, width, 3)) + rng.normal(0, 4, (height, width, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def _pole(img: np.ndarray, cx: int, y_from: int) -> None:
    cv2.rectangle(img, (cx - 5, y_from), (cx + 5, img.shape[0]), (95, 95, 95), -1)


def _center_text(img: np.ndarray, text: str, cx: int, cy: int, scale: float, color, thick: int) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def draw_stop_sign(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Red octagon with white border and 'STOP' text."""
    def octagon(radius: float) -> np.ndarray:
        return np.array([(cx + radius * math.cos(math.radians(22.5 + 45 * k)),
                          cy + radius * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)], np.int32)
    _pole(img, cx, cy + r)
    cv2.fillPoly(img, [octagon(r)], WHITE)
    cv2.fillPoly(img, [octagon(r * 0.9)], RED)
    _center_text(img, "STOP", cx, cy, r / 58.0, WHITE, max(2, r // 28))


def draw_speed_limit_sign(img: np.ndarray, cx: int, cy: int, r: int, value: str = "50") -> None:
    """White disc with a red ring and dark digits."""
    _pole(img, cx, cy + r)
    cv2.circle(img, (cx, cy), r, RED, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r * 0.76), WHITE, -1, cv2.LINE_AA)
    _center_text(img, value, cx, cy, r / 30.0, BLACK, max(2, r // 12))


def draw_no_entry_sign(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Solid red disc with a white horizontal bar."""
    _pole(img, cx, cy + r)
    cv2.circle(img, (cx, cy), r, RED, -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - int(0.65 * r), cy - int(0.13 * r)),
                  (cx + int(0.65 * r), cy + int(0.13 * r)), WHITE, -1)


def draw_warning_sign(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Red-bordered white triangle with an exclamation mark."""
    def triangle(radius: float) -> np.ndarray:
        return np.array([(cx + radius * math.cos(math.radians(a)), cy + radius * math.sin(math.radians(a)))
                         for a in (-90, 30, 150)], np.int32)
    _pole(img, cx, cy + int(r * 0.5))
    cv2.fillPoly(img, [triangle(r)], RED)
    cv2.fillPoly(img, [triangle(r * 0.58)], WHITE)
    cv2.rectangle(img, (cx - r // 20, cy - r // 4), (cx + r // 20, cy + r // 12), BLACK, -1)
    cv2.circle(img, (cx, cy + r // 5), max(2, r // 20), BLACK, -1)


def draw_direction_sign(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Blue square with a white arrow."""
    _pole(img, cx, cy + r)
    cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), BLUE, -1)
    cv2.arrowedLine(img, (cx - int(0.55 * r), cy), (cx + int(0.55 * r), cy), WHITE,
                    max(4, r // 8), tipLength=0.45)


def draw_car_with_plate(img: np.ndarray, x: int, y: int, w: int, h: int, text: str = "MP09AB1234") -> None:
    """Dark rectangular 'car body' with a white plate (black border and text) near the bottom."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 60), -1)
    cv2.rectangle(img, (x + w // 8, y + h // 8), (x + w - w // 8, y + h // 3), (95, 90, 90), -1)  # window
    plate_w, plate_h = int(w * 0.46), int(w * 0.46 / 4.3)
    px, py = x + (w - plate_w) // 2, y + int(h * 0.62)
    cv2.rectangle(img, (px, py), (px + plate_w, py + plate_h), (238, 238, 238), -1)
    cv2.rectangle(img, (px, py), (px + plate_w, py + plate_h), BLACK, 2)
    # Choose the font scale so the text fills ~88% of the plate width.
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    _center_text(img, text, px + plate_w // 2, py + plate_h // 2, 0.88 * plate_w / tw, BLACK, 2)


def _scene(seed: int, width: int = 640, height: int = 480) -> np.ndarray:
    return make_background(width, height, seed)


def build_samples() -> dict[str, np.ndarray]:
    """Return {filename: image} for all synthetic demo images."""
    samples: dict[str, np.ndarray] = {}

    img = _scene(1); draw_stop_sign(img, 320, 190, 110); samples["synthetic_stop_sign.png"] = img
    img = _scene(2); draw_speed_limit_sign(img, 320, 190, 105); samples["synthetic_speed_limit.png"] = img
    img = _scene(3); draw_no_entry_sign(img, 320, 190, 100); samples["synthetic_no_entry.png"] = img
    img = _scene(4); draw_warning_sign(img, 320, 210, 120); samples["synthetic_warning.png"] = img
    img = _scene(5); draw_direction_sign(img, 320, 190, 95); samples["synthetic_direction.png"] = img
    img = _scene(6); draw_car_with_plate(img, 120, 110, 400, 300); samples["synthetic_number_plate.png"] = img

    img = _scene(7, 960, 640)
    draw_stop_sign(img, 130, 150, 80)
    draw_speed_limit_sign(img, 330, 140, 70)
    draw_car_with_plate(img, 520, 220, 380, 330)
    samples["synthetic_road_scene.png"] = img

    samples["synthetic_empty_road.png"] = _scene(8)
    return samples


def generate_samples(output_dir: Path | str = "data/sample") -> list[Path]:
    """Write all synthetic demo images (plus a README note) to output_dir."""
    output_dir = ensure_dir(Path(output_dir))
    paths = []
    for name, image in build_samples().items():
        path = output_dir / name
        prep.save_image(path, image)
        paths.append(path)
    (output_dir / "README.md").write_text(
        "# Sample data (synthetic)\n\n"
        "These images are **synthetic demo/test images** drawn with OpenCV by "
        "`src/sample_data.py`. They are not real road photographs.\n"
        "Regenerate them with `python main.py --generate-samples`.\n", encoding="utf-8")
    return paths
