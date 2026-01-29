# Contributing to photo-quality-analyzer

Welcome! This is the **Core SDK** for Local Computer Vision.

## 🔬 Mission
This library provides the "Eyes" for the ecosystem. Our goal is to build the most accurate, privacy-first image assessment SDK in Python.

## 🛠️ Development

### 1. Installation
```bash
git clone https://github.com/yourusername/photo-quality-analyzer.git
cd photo-quality-analyzer
pip install -e .
```

### 2. Testing
We enforce strict unit testing for all physics logic.
```bash
# Run the pure physics validation suite
python3 -m unittest discover tests
```

## 🏗️ Architecture
*   **`analyzer.py`**: The monolithic core (currently).
*   **Metrics**: Implemented as pure functions (`_calculate_sharpness`, etc.).
*   **State**: The library is stateless; it analyzes one image at a time.

## 🤝 Contribution Guidelines
*   **Performance**: All metrics must be efficient (avoid heavy dependencies like TensorFlow unless necessary; we prefer YOLO/ONNX).
*   **Resolution**: Always prioritize high-res analysis (RAW previews > 1080p).
*   **Independence**: Code here must NOT depend on the `photographi` server.

---
**License**: MIT
