# Photo Quality Analyzer SDK

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/photo-quality-analyzer-core.svg)](https://pypi.org/project/photo-quality-analyzer-core/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Physics & Neural Engine behind [`photographi`](https://github.com/prasadabhishek/photo-quality-analyzer).**

`photo-quality-analyzer` is a local-first Computer Vision SDK designed to programmatically assess image quality. It bridges the gap between raw pixel data and human aesthetic perception using a hybrid approach of **Signal Processing** (physics) and **Neural Networks** (context).

---

## ⚡ Key Capabilities

*   **FFT Sharpness**: Uses Fast Fourier Transform Anisotropy to measure purely optical sharpness, invariant to rotation, with **Aperture-Aware diffraction adjustments**.
*   **Scientific Exposure**: Analyzes luminance histograms using Ansel Adams' **Zone System** and **Shannon Entropy** to detect clipping and tonal density issues.
*   **Physics-Aware Scoring**: Normalizes metrics against an expanded database of **147+ camera models** (DXOMARK/PhotonsToPhotos benchmarks).
*   **Neural Subject Detection**: Leverages **YOLOv11** to identify main subjects, ensuring metrics like focus and composition are calculated on the *subject*.
*   **RAW Fidelity**: High-fidelity extraction from ARW, CR2, NEF, DNG, and more via `rawpy` or focused EXIF preview extraction.
*   **Privacy First**: 100% local execution. No internet access or telemetry required.

---

## 📦 Installation

Primary installation is via PyPI:

```bash
pip install photo-quality-analyzer-core
```

To include RAW support (requires `LibRaw` dependencies):

```bash
pip install "photo-quality-analyzer-core[raw]"
```

Alternatively, install from source:

```bash
git clone https://github.com/prasadabhishek/photo-quality-analyzer.git
cd photo-quality-analyzer
pip install -e .
```

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
print(f"Sharpness: {metrics['sharpness']['score']}")
print(f"Description: {result['description']}")
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
exposure, explanation = _calculate_exposure(img)
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

| Metric | Technology | Scientific Basis |
| :--- | :--- | :--- |
| **Sharpness** | FFT Anisotropy | [Diffraction Limits](https://www.cambridgeincolour.com/tutorials/diffraction-photography.htm) |
| **Exposure** | Histogram Zones | [Ansel Adams Zone System](https://en.wikipedia.org/wiki/Zone_System) |
| **Subject** | YOLOv11 (Nano/Med) | [Object Detection](https://arxiv.org/abs/1506.02640) |
| **Noise** | Variance Filters | [Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Signal-to-noise_ratio) |
| **Dynamic Range** | Tonal Entropy | [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) |

---

## 🤝 Contributing

We welcome contributions! Please see `tests/` for the standalone unit test suite.

```bash
# Run the library-specific test suite
PYTHONPATH=. python3 -m unittest discover tests
```

---

**License**: MIT
