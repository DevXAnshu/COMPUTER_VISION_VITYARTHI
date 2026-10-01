"""Road Vision Analyzer - command-line entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import ocr as ocr_module
from src.analyzer import analyze_input
from src.sample_data import generate_samples
from src.utils import AnalyzerError, banner, format_batch_summary, format_image_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Road Vision Analyzer: traffic sign and vehicle number plate detection "
                    "using classical Computer Vision.")
    parser.add_argument("--input", "-i", help="Path to an image file or a folder of images.")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory (default: outputs).")
    parser.add_argument("--ocr", action="store_true",
                        help="Try to read plate text with pytesseract (optional dependency).")
    parser.add_argument("--generate-samples", action="store_true",
                        help="Create synthetic demo images in --sample-dir and exit (unless --input is given).")
    parser.add_argument("--sample-dir", default="data/sample", help="Folder for --generate-samples.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.generate_samples:
        try:
            paths = generate_samples(args.sample_dir)
        except AnalyzerError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print(f"Generated {len(paths)} synthetic demo images in {args.sample_dir}")
        if not args.input:
            return 0
    if not args.input:
        parser.error("--input is required (or use --generate-samples).")

    print(banner())
    print()
    if args.ocr:
        available, message = ocr_module.ocr_available()
        if not available:
            print(f"[WARNING] OCR requested but unavailable: {message}")
            print("          Continuing with plate detection only.\n")

    def show(result: dict) -> None:
        if result["status"] == "ok":
            print(format_image_report(result))
        else:
            print(f"Input: {result['image']}\n[ERROR] {result['error']}\n{'-' * 45}")
        print()

    try:
        outcome = analyze_input(args.input, args.output, use_ocr=args.ocr, on_result=show)
    except AnalyzerError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    summary = outcome["summary"]
    if summary["total_images"] > 1 or Path(args.input).is_dir():
        print(format_batch_summary(summary))
        print()
    print(f"JSON report saved:\n{outcome['report_path']}\n")
    if summary["processed"] == 0:
        print("Analysis failed: no image could be processed.")
        return 1
    print("Analysis completed successfully." if not summary["failed"]
          else f"Analysis completed with {summary['failed']} failed image(s).")
    print("=" * 45)
    return 0


if __name__ == "__main__":
    sys.exit(main())
