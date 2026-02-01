# Photo Quality Analyzer 📸✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/photo-quality-analyzer-core.svg)](https://pypi.org/project/photo-quality-analyzer-core/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Are you drowning in a sea of digital photos, unsure which ones truly shine? The **Photo Quality Analyzer** is your intelligent command-line companion and Python SDK, designed to meticulously evaluate your photographs based on a suite of technical metrics. Think of it as a scientifically-grounded first pass, giving you objective insights into the technical craftsmanship of your images.

Leveraging the power of **OpenCV** for signal processing and **YOLOv11** for context-aware object detection, this engine goes beyond simple checks. It identifies subjects to assess focus and composition specifically on what matters, normalized against a database of **147+ professional camera models**.

---

## ⚡ Features

The Photo Quality Analyzer dissects your images, providing scores and explanations for:

*   🎯 **Intelligent Focus Assessment**:
    *   Utilizes **YOLOv11** to detect primary subjects.
    *   Calculates sharpness specifically on these subjects, adjusted for **Aperture-Aware diffraction limits**.
*   🔪 **FFT Sharpness Analysis**: Uses Fast Fourier Transform Anisotropy to measure purely optical clarity, invariant to rotation.
*   💡 **Scientific Exposure Evaluation**: Analyzes histograms using Ansel Adams' **Zone System** and **Shannon Entropy**.
*   🔇 **Noise Estimation**: Estimates sensor ISO noise using variance filters and signal-to-noise ratios.
*   🎨 **Color & Palette Engine**: Assesses color neutrality (Neutral Pixel Selection) and generates aesthetic color palettes.
*   🌈 **Hardware-Aware Dynamic Range**: Evaluates the tonal range based on the specific sensor capabilities of your camera (e.g., Sony A1, Nikon Z9, Canon R5).

---

## 🔬 How It Works & What to Expect

This SDK provides a **technical, physics-based estimate** of image quality. It's a tool to guide and inform, bridging the gap between raw pixel data and human aesthetic perception.

| Metric | Technology | Scientific Basis |
| :--- | :--- | :--- |
| **Sharpness** | FFT Anisotropy | [Diffraction Limits](https://www.cambridgeincolour.com/tutorials/diffraction-photography.htm) |
| **Exposure** | Histogram Zones | [Ansel Adams Zone System](https://en.wikipedia.org/wiki/Zone_System) |
| **Subject** | YOLOv11 (Nano/Med) | [Object Detection](https://arxiv.org/abs/1506.02640) |
| **Noise** | Variance Filters | [Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Signal-to-noise_ratio) |
| **Dynamic Range** | Tonal Entropy | [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) |

**Hardware Intelligence:**
The engine includes a bundled database of **147+ camera models**. It automatically extracts EXIF data to normalize scores—meaning a photo from a 1-inch sensor point-and-shoot isn't unfairly compared to a 100MP Medium Format sensor.

---

## 📦 Setup & Installation

The **Photo Quality Analyzer Core** is available on PyPI.

1.  **Install via pip:**
    ```bash
    pip install photo-quality-analyzer-core
    ```
    *Note: Full RAW support (ARW, CR2, NEF, DNG) is included by default.*

2.  **Install from Source (Development):**
    ```bash
    git clone https://github.com/prasadabhishek/photo-quality-analyzer.git
    cd photo-quality-analyzer
    pip install -e .
    ```

---

## 💻 Usage

### 🚀 Command-Line Interface
If you are using the core script directly:

```bash
python analyzer.py --folder_path /path/to/your/images
```

### 🐍 Python SDK Example
The `evaluate_photo_quality` function is the main entry point for developers:

```python
from photo_quality_analyzer_core.analyzer import evaluate_photo_quality

# Analyze a single image (JPEG or RAW)
result = evaluate_photo_quality("path/to/photo.arw")

print(f"Score: {result['overallConfidence']:.2f}/1.0")
print(f"Judgement: {result['judgement']}") 
print(f"Description: {result['judgement_description']}")

# Access granular signal data
print(f"FFT Sharpness: {result['metrics']['sharpness']['score']}")
```

---

## 🤝 Contributing & Feedback

We welcome contributions! Please feel free to submit pull requests or open issues. To run the internal test suite:

```bash
PYTHONPATH=. python3 -m unittest discover tests
```

---

## 📜 License

The Photo Quality Analyzer script and its associated documentation (excluding third-party dependencies) are released under the [MIT License](LICENSE).

**Third-Party Licenses:**
*   This project utilizes the `ultralytics` package, licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
*   YOLO models (e.g., `yolov11n.pt`) are typically also licensed under AGPL-3.0 if obtained from Ultralytics.
