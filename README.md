# Photo Quality Analyzer SDK

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Physics & Neural Engine behind [`photographi`](../photographi).**

`photo-quality-analyzer` is a local-first Computer Vision SDK designed to programmatically assess image quality. It bridges the gap between raw pixel data and human aesthetic perception using a hybrid approach of **Signal Processing** (physics) and **Neural Networks** (context).

---

## ⚡ Key Capabilities

*   **FFT Sharpness**: Uses Fast Fourier Transform (magnitude spectrum moments) to measure purely optical sharpness, invariant to rotation.
*   **Zone System Exposure**: Analyzes luminance histograms using Ansel Adams' Zone System to detect clipping and dynamic range issues.
*   **Neural Subject Detection**: Leverages **YOLO12x** to identify main subjects, ensuring metrics are calculated on the *subject*, not the background.
*   **RAW Fidelity**: Extracts high-resolution (>1080p) previews from Sony ARW, Canon CR2, and Nikon NEF files for accurate analysis.
*   **Privacy First**: 100% local execution. No images ever leave your machine.

---

## 📦 Installation

```bash
# Install from source (for now)
git clone https://github.com/yourusername/photo-quality-analyzer.git
cd photo-quality-analyzer
pip install -e .
```

*Note: Requires `opencv-python`, `numpy`, `scipy`, `rawpy`, and `ultralytics`.*

---

## 💻 Python API Usage

### 1. High-Level Evaluation
The `evaluate_photo_quality` function is the main entry point, returning a composite score and human-readable judgement.

```python
from photo_quality_analyzer_core.analyzer import evaluate_photo_quality

# Analyze a single image
result = evaluate_photo_quality("path/to/photo.jpg")

print(f"Score: {result['overallConfidence']:.2f}/1.0")
print(f"Judgement: {result['judgement']}")  # e.g., "Sharp & Well-Exposed"

# Access granular metrics
metrics = result['metrics']
print(f"Sharpness: {metrics['sharpness_score']}")
print(f"Main Subject: {result.get('main_subject_name', 'None')}")
```

### 2. Low-Level Signal Processing
Access specific physics modules for granular analysis.

```python
import cv2
from photo_quality_analyzer_core.analyzer import _calculate_sharpness, _calculate_exposure

img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)

# Get pure FFT sharpness score
sharpness, explanation = _calculate_sharpness(img)
print(f"FFT Sharpness: {sharpness:.4f} ({explanation})")

# Check Ansel Adams zones
exposure, zone_data = _calculate_exposure(img)
print(f"Exposure Score: {exposure}")
```

### 3. RAW Pipeline
Currently supports **Sony ARW**, **Canon CR2**, and others via `rawpy` or focused EXIF preview extraction.

```python
from photo_quality_analyzer_core.analyzer import _load_image_with_raw_support

# Automatically extracts best available preview from RAW
img = _load_image_with_raw_support("DSC001.ARW")
# Returns a standard OpenCV BGR array
```

---

## 🔬 How It Works

| Metric | Technology | Why it matters |
| :--- | :--- | :--- |
| **Sharpness** | FFT (Fast Fourier Transform) | Distinguishes between "artistic bokeh" and "missed focus" by analyzing high-frequency energy. |
| **Exposure** | Histogram Zones | Penalizes only "destructive" clipping (blown highlights) while allowing artistic shadows. |
| **Subject** | YOLO12x (Nano/X) | Understands *what* is in the photo to weight sharpness on the subject (e.g., eyes) over the background. |
| **Noise** | Variance Filters | Estimates sensor ISO noise to penalize low-light grain heavily. |

---

## 🤝 Contributing

We welcome contributions! Please see `tests/` for the standalone unit test suite.

```bash
# Run the library-specific test suite
python3 -m unittest discover tests
```

---

**License**: MIT
