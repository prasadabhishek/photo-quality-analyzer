# photo-quality-analyzer

> Intelligent technical assessment for digital photography.

[photo-quality-analyzer](https://github.com/prasadabhishek/photo-quality-analyzer) is a local-first Python SDK and CLI tool that uses signal processing and computer vision to objectively score photographic quality. It normalizes metrics against a database of **147+ camera models** to account for sensor-specific physics like diffraction limits and dynamic range baselines.

### From PyPI
```bash
pip install photo-quality-analyzer-core
```

### From GitHub (Source)
For developers or users who want the latest changes directly from the source:
```bash
pip install git+https://github.com/prasadabhishek/photo-quality-analyzer.git
```

---

## What This Library Is (And Isn't)

This library is a **Technical Quality Filter** ("Janitor"), not an artistic evaluator ("Curator").

### ✅ What it WILL do:
- Identify out-of-focus, underexposed, or noisy images
- Filter out broken shots from large photo libraries (10,000+ images)
- Provide objective technical metrics (sharpness, noise, dynamic range)
- Normalize scores against known camera sensor physics

### ❌ What it WON'T do:
- Judge artistic merit or emotional impact
- Understand intentional creative choices (low-key lighting, film grain, etc.)
- Replace human curation for portfolio selection
- Prefer "interesting" photos over "boring but technically perfect" ones

**Use Case**: Wedding photographers culling 5,000 shots to eliminate camera-shake blurs and exposure failures—not fine art curation.

---

## Metrics

The engine evaluates technical quality through a multi-dimensional lens:

- **Sharpness**: FFT-based acutance, invariant to rotation and noise.
- **Exposure**: Ansel Adams Zone System analysis for clipping detection.
- **Focus**: ROI-specific sharpness on the main subject (auto-detected).
- **Noise**: Statistical variance estimation for ISO-related grain.
- **Dynamic Range**: Tonal entropy and sensor-aware potential.
- **Color Balance**: Neutral pixel selection for finding color casts.

- **Color Balance**: Neutral pixel selection for finding color casts.

For more information, see our documentation:
- 📖 **[USAGE.md](docs/USAGE.md)**: Practical examples and CLI guides.
- ⚙️ **[API.md](docs/API.md)**: Technical reference for Python developers.
- 🔬 **[SCIENCE.md](docs/SCIENCE.md)**: Deep dive into the underlying physics and algorithms.

## Usage

### CLI
Analyze an entire folder and optionally move files based on quality:

```bash
python analyzer.py --folder_path /path/to/photos --move
```

### SDK
```python
from photo_quality_analyzer_core.analyzer import evaluate_photo_quality

# Works with JPEGs and RAW files
result = evaluate_photo_quality("photo.arw")
print(result['judgement']) # "Excellent", "Good", etc.
```

*See [USAGE.md](docs/USAGE.md) for more advanced examples (AI toggling, metric filtering, etc).*

## How it works

The engine uses a hybrid approach to distinguish between artistic intent and technical failure:

1.  **FFT Anisotropy**: Measures purely optical acutance, invariant to rotation. Adjusted for **Aperture-aware diffraction**.
2.  **Zone System Histogram**: Analyzes luminance using Ansel Adams' Zone System to detect destructive clipping.
3.  **Neural ROI (YOLO26)**: Leverages the latest **YOLO26** (January 2026 release) via **ONNX Runtime** to identify main subjects, ensuring metrics are calculated on the subject rather than the background.
4.  **Sensor Normalization**: Benchmarks images against the known limits of the specific camera sensor (Full Frame vs APS-C vs 1-inch).

## Technology Stack

- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)**: Optimized, lightweight inference engine (replaced PyTorch).
- **[YOLO26](https://github.com/ultralytics/ultralytics)**: Transformer-based subject detection (43% faster on CPUs).
- **OpenCV (Headless)**: Efficient image processing without GUI overhead.

### `evaluate_photo_quality(file_path, ...)`
The primary entry point. It returns a dictionary containing scores, qualitative labels, and AI-generated scene descriptions.

*See [API.md](docs/API.md) for full function signatures and return types.*

## Contributing

Contributions are welcome! Please run the test suite before submitting:

```bash
PYTHONPATH=. python3 -m unittest discover tests
```

## License

[MIT](LICENSE)
