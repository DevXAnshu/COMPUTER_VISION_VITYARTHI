"""Tests for the traffic sign module (synthetic images only, no network)."""
import numpy as np
import pytest

from src.sample_data import build_samples
from src.traffic_sign import SIGN_TYPES, detect_traffic_signs, find_sign_candidates

SAMPLES = build_samples()


def test_candidates_found_for_stop_sign():
    candidates = find_sign_candidates(SAMPLES["synthetic_stop_sign.png"])
    assert len(candidates) >= 1
    assert candidates[0]["color"] == "red"


@pytest.mark.parametrize("name, expected_type, expected_shape", [
    ("synthetic_stop_sign.png", "STOP", "Octagon"),
    ("synthetic_speed_limit.png", "SPEED_LIMIT", "Circle"),
    ("synthetic_no_entry.png", "NO_ENTRY", "Circle"),
    ("synthetic_warning.png", "WARNING", "Triangle"),
    ("synthetic_direction.png", "DIRECTION", "Square"),
])
def test_classification_of_synthetic_signs(name, expected_type, expected_shape):
    signs = detect_traffic_signs(SAMPLES[name])
    assert len(signs) == 1
    assert signs[0]["type"] == expected_type
    assert signs[0]["shape"] == expected_shape


def test_sign_result_structure_and_confidence_range():
    sign = detect_traffic_signs(SAMPLES["synthetic_stop_sign.png"])[0]
    assert {"type", "shape", "color", "confidence", "bounding_box"} <= set(sign)
    assert sign["type"] in SIGN_TYPES
    assert 0.0 <= sign["confidence"] <= 100.0
    assert set(sign["bounding_box"]) == {"x", "y", "width", "height"}


def test_no_signs_in_empty_image():
    assert detect_traffic_signs(SAMPLES["synthetic_empty_road.png"]) == []
    assert detect_traffic_signs(np.zeros((100, 100, 3), np.uint8)) == []
