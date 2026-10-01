"""Tests for the number plate module (synthetic images only, no network)."""
import numpy as np

from src.number_plate import crop_plate, detect_number_plates, find_plate_candidates
from src.sample_data import build_samples

SAMPLES = build_samples()


def test_plate_candidates_exist():
    assert len(find_plate_candidates(SAMPLES["synthetic_number_plate.png"])) >= 1


def test_plate_detected_with_valid_structure():
    plates = detect_number_plates(SAMPLES["synthetic_number_plate.png"])
    assert len(plates) >= 1
    plate = plates[0]
    assert plate["detected"] is True
    assert 0.0 <= plate["confidence"] <= 100.0
    assert 2.0 <= plate["aspect_ratio"] <= 6.5
    assert set(plate["bounding_box"]) == {"x", "y", "width", "height"}
    assert plate["area"] == plate["bounding_box"]["width"] * plate["bounding_box"]["height"]


def test_plate_location_is_near_drawn_plate():
    # The plate was drawn at roughly x=228..412, y=296..339 in the synthetic image.
    box = detect_number_plates(SAMPLES["synthetic_number_plate.png"])[0]["bounding_box"]
    assert abs(box["x"] - 228) < 15 and abs(box["y"] - 296) < 15


def test_crop_plate_returns_image():
    image = SAMPLES["synthetic_number_plate.png"]
    plate = detect_number_plates(image)[0]
    crop = crop_plate(image, plate)
    assert crop.ndim == 3 and crop.shape[0] > 0 and crop.shape[1] > crop.shape[0]


def test_no_plate_in_empty_image():
    assert detect_number_plates(SAMPLES["synthetic_empty_road.png"]) == []
    assert detect_number_plates(np.zeros((100, 100, 3), np.uint8)) == []
