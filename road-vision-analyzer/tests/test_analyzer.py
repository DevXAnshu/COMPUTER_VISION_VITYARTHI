"""Tests for the analyzer, batch processing, error handling and the CLI."""
import json

import pytest

from main import main
from src.analyzer import analyze_image, analyze_input
from src.sample_data import generate_samples
from src.utils import (AnalyzerError, ImageLoadError, InputNotFoundError, NoImagesFoundError)


@pytest.fixture()
def sample_dir(tmp_path):
    folder = tmp_path / "samples"
    generate_samples(folder)
    return folder


def test_complete_result_structure(sample_dir, tmp_path):
    result = analyze_image(sample_dir / "synthetic_road_scene.png", tmp_path / "out")
    assert result["status"] == "ok"
    assert {"image", "traffic_signs", "number_plates", "annotated_image"} <= set(result)
    assert {s["type"] for s in result["traffic_signs"]} == {"STOP", "SPEED_LIMIT"}
    assert len(result["number_plates"]) == 1
    assert result["number_plates"][0]["ocr_status"] == "not_requested"
    assert (tmp_path / "out" / "detections" / "synthetic_road_scene_annotated.jpg").is_file()


def test_missing_image_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        analyze_image(tmp_path / "nope.png", tmp_path)
    with pytest.raises(InputNotFoundError):
        analyze_input(tmp_path / "nope.png", tmp_path)


def test_invalid_and_corrupted_image(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(ImageLoadError):
        analyze_image(bad, tmp_path / "out")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        analyze_image(empty, tmp_path / "out")


def test_unsupported_file_and_empty_folder(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(AnalyzerError):
        analyze_input(txt, tmp_path / "out")
    folder = tmp_path / "only_text"
    folder.mkdir()
    (folder / "a.txt").write_text("x")
    with pytest.raises(NoImagesFoundError):
        analyze_input(folder, tmp_path / "out")


def test_batch_processing_and_report(sample_dir, tmp_path):
    (sample_dir / "corrupt.jpg").write_bytes(b"garbage")  # one failing image must not stop the batch
    out = tmp_path / "out"
    outcome = analyze_input(sample_dir, out)
    summary = outcome["summary"]
    assert summary["total_images"] == 9
    assert summary["processed"] == 8 and summary["failed"] == 1
    assert summary["traffic_signs"] == 7 and summary["number_plates"] == 2
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert len(report) == 9
    assert any(r["status"] == "error" for r in report)


def test_ocr_unavailable_never_fabricates_text(sample_dir, tmp_path, monkeypatch):
    from src import ocr
    monkeypatch.setattr(ocr, "ocr_available", lambda: (False, "mocked"))
    plate = analyze_image(sample_dir / "synthetic_number_plate.png", tmp_path, use_ocr=True)["number_plates"][0]
    assert plate["ocr_text"] is None and plate["ocr_status"] == "unavailable"


def test_cli_success_and_failure(sample_dir, tmp_path, capsys):
    assert main(["--input", str(sample_dir), "--output", str(tmp_path / "o")]) == 0
    assert "BATCH SUMMARY" in capsys.readouterr().out
    assert main(["--input", str(tmp_path / "missing.jpg"), "--output", str(tmp_path / "o")]) == 1
    assert "[ERROR]" in capsys.readouterr().err
