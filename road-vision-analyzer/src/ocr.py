"""Optional OCR for number plates. The rest of the project works without this module's dependencies."""
from __future__ import annotations

import re
from functools import lru_cache

import cv2
import numpy as np

from . import preprocessing as prep

OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@lru_cache(maxsize=1)
def ocr_available() -> tuple[bool, str]:
    """Return (available, message). Checks for both pytesseract and the Tesseract binary."""
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract is not installed (pip install pytesseract)."

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False, "Tesseract OCR engine not found at configured path."

    return True, "OCR available."


def preprocess_plate_for_ocr(plate: np.ndarray) -> np.ndarray:
    """Grayscale -> upscale -> denoise -> Otsu threshold -> morphological cleanup."""
    gray = prep.to_gray(plate)
    scale = max(1.0, 80.0 / max(1, gray.shape[0]))
    gray = cv2.resize(
        gray,
        None,
        fx=scale * 2,
        fy=scale * 2,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)
    binary = prep.otsu_threshold(gray)

    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    binary = prep.morph_open(binary, 2)

    return cv2.copyMakeBorder(
        binary,
        10,
        10,
        10,
        10,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def read_plate_text(plate: np.ndarray) -> dict:
    """Run OCR on a cropped plate. Never fabricates text."""
    available, _ = ocr_available()

    if not available:
        return {"ocr_text": None, "ocr_status": "unavailable"}

    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    try:
        raw = pytesseract.image_to_string(
            preprocess_plate_for_ocr(plate),
            config=OCR_CONFIG,
        )
    except Exception:
        return {"ocr_text": None, "ocr_status": "error"}

    text = re.sub(r"[^A-Z0-9]", "", raw.upper())

    if not text:
        return {"ocr_text": None, "ocr_status": "no_text"}

    return {"ocr_text": text, "ocr_status": "ok"}
