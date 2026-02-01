# photo-quality-analyzer

> Intelligent technical assessment for digital photography.

[photo-quality-analyzer](https://github.com/prasadabhishek/photo-quality-analyzer) is a local-first Python SDK and CLI tool that uses signal processing and computer vision to objectively score photographic quality. It normalizes metrics against a database of **147+ camera models** to account for sensor-specific physics like diffraction limits and dynamic range baselines.

## Install

```bash
pip install photo-quality-analyzer-core
```
*Full RAW support (`.ARW`, `.CR2`, `.NEF`, etc.) is included by default.*

## Metrics

The engine evaluates technical quality through a multi-dimensional lens:

- **Sharpness**: FFT-based acutance, invariant to rotation and noise.
- **Exposure**: Ansel Adams Zone System analysis for clipping detection.
- **Focus**: ROI-specific sharpness on the main subject (auto-detected).
- **Noise**: Statistical variance estimation for ISO-related grain.
- **Dynamic Range**: Tonal entropy and sensor-aware potential.
- **Color Balance**: Neutral pixel selection for finding color casts.

For a deep dive into the underlying physics and signal processing, see [SCIENCE.md](docs/SCIENCE.md).

## Usage

### CLI
Analyze an entire folder and optionally move files based on quality:

```bash
python analyzer.py --folder_path /path/to/photos --move
```

### SDK
Use the high-level evaluation engine in your own Python projects:

```python
from photo_quality_analyzer_core.analyzer import evaluate_photo_quality

# Works with JPEGs and RAW files
result = evaluate_photo_quality("photo.arw")

print(result['overallConfidence'])    # 0.0 - 1.0 score
print(result['judgement'])            # "Excellent", "Good", etc.
print(result['metrics']['sharpness']) # Detailed signal data
```

## How it works

The engine uses a hybrid approach to distinguish between artistic intent and technical failure:

1.  **FFT Anisotropy**: Measures purely optical acutance, invariant to rotation. Adjusted for **Aperture-aware diffraction**.
2.  **Zone System Histogram**: Analyzes luminance using Ansel Adams' Zone System to detect destructive clipping.
3.  **Neural ROI**: Leverages **YOLOv11** to identify main subjects, ensuring metrics are calculated on the subject rather than the background.
4.  **Sensor Normalization**: Benchmarks images against the known limits of the specific camera sensor (Full Frame vs APS-C vs 1-inch).

## API

### `evaluate_photo_quality(file_path, enable_subject_detection=True)`

Returns a dictionary containing:
- `overallConfidence`: Weighted average (0-1).
- `judgement`: Qualitative label.
- `metrics`: Granular data for Sharpness, Exposure, Noise, and Dynamic Range.
- `image_description`: List of detected objects.

## Contributing

Contributions are welcome! Please run the test suite before submitting:

```bash
PYTHONPATH=. python3 -m unittest discover tests
```

## License

[MIT](LICENSE)
