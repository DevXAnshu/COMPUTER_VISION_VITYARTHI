"""High-level orchestration: analyse one image or a folder and write all outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import ocr as ocr_module
from . import preprocessing as prep
from .config import MAX_PROCESS_DIMENSION
from .detection import combine_results, draw_annotations, remove_plates_inside_signs
from .number_plate import crop_plate, detect_number_plates
from .traffic_sign import detect_traffic_signs
from .utils import AnalyzerError, collect_images, ensure_dir, save_json


def analyze_image(image_path: Path | str, output_dir: Path | str = "outputs",
                  use_ocr: bool = False, save_outputs: bool = True) -> dict:
    """Analyse one image and (optionally) save the annotated image and plate crops.

    Raises ImageLoadError for missing/invalid images and OutputError for write problems.
    Coordinates in the result refer to the processed image (longest side <= MAX_PROCESS_DIMENSION).
    """
    image_path, output_dir = Path(image_path), Path(output_dir)
    original = prep.load_image(image_path)
    image, _ = prep.resize_to_max_dimension(original, MAX_PROCESS_DIMENSION)
    height, width = image.shape[:2]

    signs = detect_traffic_signs(image)
    plates = remove_plates_inside_signs(detect_number_plates(image), signs)

    detections_dir = plates_dir = None
    if save_outputs:
        detections_dir = ensure_dir(output_dir / "detections")
        plates_dir = ensure_dir(output_dir / "plates") if plates else None

    for index, plate in enumerate(plates, 1):
        crop = crop_plate(image, plate)
        if plates_dir is not None:
            crop_path = plates_dir / f"{image_path.stem}_plate{index}.png"
            prep.save_image(crop_path, crop)
            plate["plate_image"] = crop_path.as_posix()
        if use_ocr:
            plate.update(ocr_module.read_plate_text(crop))
        else:
            plate.update({"ocr_text": None, "ocr_status": "not_requested"})

    result = combine_results(image_path.name, (width, height), signs, plates)
    result["annotated_image"] = None
    if detections_dir is not None:
        annotated_path = detections_dir / f"{image_path.stem}_annotated.jpg"
        prep.save_image(annotated_path, draw_annotations(image, signs, plates))
        result["annotated_image"] = annotated_path.as_posix()
    return result


def analyze_input(input_path: Path | str, output_dir: Path | str = "outputs", use_ocr: bool = False,
                  on_result: Callable[[dict], None] | None = None) -> dict:
    """Analyse a single image or every supported image in a folder; write report.json.

    Individual image failures are recorded and do not stop the batch. Raises AnalyzerError
    for problems that make the whole run impossible (missing input, empty folder, bad output dir).
    """
    images = collect_images(Path(input_path))
    output_dir = ensure_dir(Path(output_dir))

    results: list[dict] = []
    failed: dict[str, str] = {}
    for path in images:
        try:
            result = analyze_image(path, output_dir, use_ocr=use_ocr)
        except AnalyzerError as exc:
            failed[path.name] = str(exc)
            result = {"image": path.name, "status": "error", "error": str(exc),
                      "traffic_signs": [], "number_plates": [], "annotated_image": None}
        results.append(result)
        if on_result:
            on_result(result)

    ok = [r for r in results if r["status"] == "ok"]
    summary = {
        "total_images": len(images),
        "processed": len(ok),
        "failed": len(failed),
        "traffic_signs": sum(len(r["traffic_signs"]) for r in ok),
        "number_plates": sum(len(r["number_plates"]) for r in ok),
        "failed_images": failed,
    }
    report_path = output_dir / "report.json"
    save_json(results, report_path)
    return {"results": results, "summary": summary, "report_path": report_path.as_posix()}
