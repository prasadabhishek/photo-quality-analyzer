# API Reference

This document outlines the public API of the `photo-quality-analyzer` package.

---

## 📦 Core Module

### `evaluate_photo_quality`

The main entry point for analyzing an image. It handles loading, signal processing, neural inference, and judgement fusion.

```python
def evaluate_photo_quality(
    image_path: str,
    requested_metrics: list[str] = None,
    enable_subject_detection: bool = True,
    model_size: str = "nano"
) -> dict
```

**Parameters**:
- `image_path` (str): Path to the target image (JPEG, PNG, RAW).
- `requested_metrics` (list[str], optional): List of metrics to calculate (e.g., `["sharpness", "exposure"]`). Defaults to `"all"`.
- `enable_subject_detection` (bool): If `True`, runs YOLO26 for ROI-based analysis. Defaults to `True`.
- `model_size` (str): YOLO model size (`"nano"`). Defaults to `"nano"`.

**Returns**:
A `dict` with the following schema:
```json
{
  "overallConfidence": float,    // Weighted score (0.0 - 1.0)
  "technicalScore": float,       // Physics-based score
  "aestheticScore": float,       // Composition-based score
  "judgement": string,          // "Excellent", "Good", "Fair", "Poor"
  "judgementDescription": str,  // Paragraph explaining the score
  "description": string,        // AI summary of scene content
  "metrics": {                  // Detailed breakdown
     "sharpness": { "score": float, "explanation": str },
     "exposure": { "score": float, "explanation": str },
     "noise": { "score": float, "explanation": str },
     "focus": { "score": float, "explanation": str },
     "color": { "score": float, "explanation": str },
     "dynamicRange": { "score": float, "explanation": str },
     "composition": { "score": float, "explanation": str }
  },
  "metadataStatus": string      // "present" or "missing"
}
```

---

## 🔬 Signal Processing Primitives

Low-level functions for granular analysis. Use these if you are building your own processing pipeline.

### `_calculate_sharpness`
```python
def _calculate_sharpness(gray_img: np.ndarray, metadata: dict = None) -> tuple[float, str]
```
Calculates sharpness scores using FFT Anisotropy.

### `_calculate_exposure`
```python
def _calculate_exposure(gray_img: np.ndarray, metadata: dict = None) -> tuple[float, str]
```
Evaluates exposure based on the Zone System.

### `_calculate_noise`
```python
def _calculate_noise(gray_img: np.ndarray, metadata: dict = None) -> tuple[float, str]
```
Estimates sensor noise variance.

---

## 🛠️ Utility Functions

### `create_xmp_sidecar`
```python
def create_xmp_sidecar(image_path: str, judgement: str, confidence: float) -> None
```
Generates an Adobe-compatible XMP sidecar.
- **Rating**: Set from 0 to 5 stars based on `confidence`.
- **Label**: Set to "Rejected" if `confidence < 0.4`.

### `generate_color_palette`
```python
def generate_color_palette(image_path: str, num_colors: int = 5) -> list[str]
```
Extracts dominant colors using K-Means clustering.
- **Returns**: List of Hex color codes (e.g., `["#FFFFFF", "#000000"]`).
