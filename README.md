# Road Vision Analyzer

A computer vision application for detecting **traffic signs** and **vehicle number plates** from road images, with optional **OCR-based number plate recognition**.

The project uses OpenCV-based image processing and detection techniques, with Tesseract OCR for extracting alphanumeric text from detected number plates.

---

## Features

* Traffic sign detection
* Traffic sign classification
* Traffic sign shape detection
* Traffic sign color analysis
* Vehicle number plate detection
* Number plate confidence scoring
* Number plate aspect-ratio analysis
* OCR-based number plate text recognition
* Automatic image annotation
* JSON report generation
* Command-line interface
* Automated test suite
* Optional OCR dependency
* Works on Windows with Python virtual environments

---

## Project Overview

Road Vision Analyzer processes road images and identifies important visual elements.

The application currently supports two primary detection tasks:

### Traffic Sign Detection

The system can detect traffic signs and provide information such as:

* Sign type
* Shape
* Dominant color
* Confidence
* Bounding-box location

Example:

```text
TRAFFIC SIGNS
---------------------------------------------
Detected: 1

[1]
Type       : WARNING
Shape      : Triangle
Color      : Red
Confidence : 98%
Location   : (218, 92, 204, 180)
```

### Number Plate Detection

The system detects vehicle number plates and can optionally extract the text using OCR.

Example:

```text
NUMBER PLATES
---------------------------------------------
Detected: 1

[1]
Confidence   : 96%
Location     : (168, 163, 140, 35)
Aspect Ratio : 4.00
OCR          : 435SR
```

---

## Technology Stack

| Technology    | Purpose                              |
| ------------- | ------------------------------------ |
| Python        | Core programming language            |
| OpenCV        | Computer vision and image processing |
| NumPy         | Numerical image operations           |
| Pillow        | Image processing support             |
| PyTesseract   | Python interface for Tesseract OCR   |
| Tesseract OCR | Number plate text recognition        |
| Pytest        | Automated testing                    |

---

## Project Structure

```text
road-vision-analyzer/
│
├── data/
│   └── sample/
│       ├── synthetic_warning.png
│       └── synthetic_number_plate_plate1.png
│
├── outputs/
│   ├── detections/
│   │   └── *_annotated.jpg
│   │
│   └── report.json
│
├── src/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── number_plate.py
│   ├── ocr.py
│   ├── preprocessing.py
│   ├── traffic_sign.py
│   └── utils.py
│
├── tests/
│   ├── test_analyzer.py
│   ├── test_number_plate.py
│   └── test_traffic_sign.py
│
├── main.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Requirements

Before installing the project, make sure you have:

* Python 3.10 or newer
* pip
* Windows, Linux, or macOS
* Tesseract OCR if OCR functionality is required

Python dependencies are listed in:

```text
requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd road-vision-analyzer
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```text
(.venv) PS C:\...\road-vision-analyzer>
```

### 4. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# OCR Setup

Number plate detection works independently from OCR.

OCR requires the following two components:

```text
PyTesseract
    +
Tesseract OCR Engine
```

Installing `pytesseract` alone is not enough.

## Install Tesseract

Install Tesseract OCR for your operating system.

On Windows, the executable is commonly installed at:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

The project is configured to use this path.

In:

```text
src/ocr.py
```

the configured executable path is:

```python
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

If Tesseract is installed somewhere else, update this path.

---

## Verify Tesseract

From PowerShell:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

Example:

```text
tesseract v5.5.3
```

You can also verify that Python can communicate with Tesseract:

```powershell
python -c "import pytesseract; pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'; print(pytesseract.get_tesseract_version())"
```

Expected output:

```text
5.5.3
```

---

# Usage

## Analyze an Image

Run the analyzer without OCR:

```powershell
python main.py --input "data\sample\synthetic_warning.png"
```

The program analyzes the image and generates an annotated output image and JSON report.

---

## Analyze a Number Plate with OCR

Use the `--ocr` option:

```powershell
python main.py --input "data\sample\synthetic_number_plate_plate1.png" --ocr
```

Example output:

```text
=============================================
             ROAD VISION ANALYZER
=============================================

Input: synthetic_number_plate_plate1.png

TRAFFIC SIGNS
---------------------------------------------
Detected: 0
(no traffic signs found)

NUMBER PLATES
---------------------------------------------
Detected: 1

[1]
Confidence   : 96%
Location     : (168, 163, 140, 35)
Aspect Ratio : 4.00
OCR          : 435SR

---------------------------------------------
Annotated image saved:
outputs/detections/synthetic_number_plate_plate1_annotated.jpg

JSON report saved:
outputs/report.json

Analysis completed successfully.
=============================================
```

---

# Command-Line Options

Basic analysis:

```powershell
python main.py --input <IMAGE_PATH>
```

Analysis with OCR:

```powershell
python main.py --input <IMAGE_PATH> --ocr
```

Example:

```powershell
python main.py --input "data\sample\synthetic_number_plate_plate1.png" --ocr
```

---

# Output

The application generates two primary types of output.

## Annotated Image

Annotated images are stored in:

```text
outputs/detections/
```

Example:

```text
outputs/detections/synthetic_number_plate_plate1_annotated.jpg
```

The annotated image contains visual information about detected objects and their locations.

---

## JSON Report

The analysis report is stored at:

```text
outputs/report.json
```

The report contains structured detection information that can be consumed by other applications or used for further analysis.

A simplified example:

```json
{
  "input": "synthetic_number_plate_plate1.png",
  "number_plates": [
    {
      "confidence": 0.96,
      "location": [168, 163, 140, 35],
      "aspect_ratio": 4.0,
      "ocr_text": "435SR"
    }
  ]
}
```

---

# OCR Processing Pipeline

The number plate OCR pipeline follows these steps:

```text
Input Image
     |
     v
Number Plate Detection
     |
     v
Plate Cropping
     |
     v
Grayscale Conversion
     |
     v
Image Upscaling
     |
     v
Noise Reduction
     |
     v
Otsu Thresholding
     |
     v
Morphological Processing
     |
     v
Tesseract OCR
     |
     v
Text Cleaning
     |
     v
Recognized Plate Text
```

The OCR module restricts recognized characters to:

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
```

This helps reduce unwanted characters in number plate results.

---

# Architecture

The application is organized into separate modules.

```text
main.py
   |
   v
Analyzer
   |
   +--------------------+
   |                    |
   v                    v
Traffic Sign        Number Plate
Detection           Detection
   |                    |
   v                    v
Classification        Plate Crop
   |                    |
   |                    v
   |                  OCR
   |                    |
   +---------+----------+
             |
             v
       Result Generation
             |
       +-----+------+
       |            |
       v            v
 Annotated       JSON
   Image         Report
```

---

# Testing

The project includes an automated test suite using Pytest.

Run all tests:

```powershell
python -m pytest
```

Current test result:

```text
================================================ test session starts =================================================
platform win32 -- Python 3.13.5
collected 20 items

tests\test_analyzer.py .......
tests\test_number_plate.py .....
tests\test_traffic_sign.py ........

================================================= 20 passed =================================================
```

The current project contains:

```text
20 automated tests
20 passing
0 failing
```

---

# Test Coverage

The test suite covers:

### Analyzer

```text
tests/test_analyzer.py
```

Tests the main analysis pipeline and integration behavior.

### Number Plate Detection

```text
tests/test_number_plate.py
```

Tests number plate detection, geometry, confidence handling, and related processing.

### Traffic Sign Detection

```text
tests/test_traffic_sign.py
```

Tests traffic sign detection and classification behavior.

---

# Error Handling

The application is designed to continue operating when optional OCR functionality is unavailable.

For example, if Tesseract is not installed, number plate detection can still operate:

```text
Number Plate Detection
        |
        v
      Works
        |
        +----> OCR unavailable
```

This separation keeps OCR optional rather than making it a requirement for the core detection pipeline.

---

# Troubleshooting

## Tesseract OCR Engine Not Found

If you see:

```text
OCR: Unavailable
```

verify that Tesseract is installed:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

Then verify the configured path in:

```text
src/ocr.py
```

Look for:

```python
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## `pytesseract` Not Installed

Install it inside the active virtual environment:

```powershell
pip install pytesseract
```

Then verify:

```powershell
python -c "import pytesseract; print(pytesseract.__version__)"
```

---

## Tests Not Running

Make sure the virtual environment is activated:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```powershell
python -m pytest
```

---

## PowerShell Execution Policy

If PowerShell prevents virtual environment activation, you may need to allow local scripts for your user account:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# Development

To run the project during development:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py --input "data\sample\synthetic_number_plate_plate1.png" --ocr
```

Before committing changes, run:

```powershell
python -m pytest
```

A successful development build should maintain:

```text
20 passed
```

---

# Future Improvements

Potential future enhancements include:

* Real-time webcam processing
* Video file analysis
* Improved number plate detection
* Multi-country license plate support
* More advanced OCR preprocessing
* Character-level OCR confidence
* Vehicle detection
* Vehicle tracking
* Speed estimation
* Automatic plate database integration
* CSV export
* Web-based dashboard
* REST API
* GPU acceleration
* Deep-learning-based object detection
* Improved traffic sign classification
* Additional traffic sign categories

---

# Limitations

The current implementation is primarily intended for image-based analysis.

OCR accuracy can vary depending on:

* Image resolution
* Lighting conditions
* Plate angle
* Motion blur
* Plate size
* Image noise
* Character font
* Occlusion
* Camera quality

OCR results should therefore be treated as computer-vision estimates rather than guaranteed ground truth.

---

# Example Workflow

A typical workflow looks like this:

```text
1. Provide road image
        |
        v
2. Run Road Vision Analyzer
        |
        v
3. Detect traffic signs
        |
        v
4. Detect number plates
        |
        v
5. Crop detected plates
        |
        v
6. Run OCR
        |
        v
7. Generate annotated image
        |
        v
8. Generate JSON report
```

Example command:

```powershell
python main.py --input "data\sample\synthetic_number_plate_plate1.png" --ocr
```

---

# Project Status

The current implementation successfully supports:

```text
Traffic Sign Detection       [✓]
Traffic Sign Classification  [✓]
Number Plate Detection       [✓]
Number Plate OCR             [✓]
Annotated Images             [✓]
JSON Reporting               [✓]
Automated Testing            [✓]
```

Current automated test status:

```text
20 / 20 tests passing
```

---

# License

Add your preferred license here.

For example:

```text
MIT License
```

If this project is intended for academic submission, use the license or usage terms required by your institution or project specification.

---

# Author

**BIKRAM KUMAR 24BAI10471**

Road Vision Analyzer — Computer Vision Project

---

# Acknowledgements

This project uses open-source technologies including:

* Python
* OpenCV
* NumPy
* Pillow
* PyTesseract
* Tesseract OCR
* Pytest
