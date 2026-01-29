# API Reference

This document outlines the public API of the `photo-quality-analyzer` package.

## 📦 Core Module

### `evaluate_photo_quality`

The main entry point for analyzing an image. It handles loading, signal processing, neural inference, and judgement fusion.

```python
def evaluate_photo_quality(
    image_path: str, 
    enable_subject_detection: bool = True
) -> dict
```

**Parameters**:
*   `image_path` (str): Absolute or relative path to the image file. Supports `.jpg`, `.png`, and RAW formats (`.arw`, `.cr2`) if `rawpy` is installed.
*   `enable_subject_detection` (bool): If `True`, runs YOLO ID detection (slower, ~1.5s). If `False`, runs pure physics analysis (~0.05s).

**Returns**:
A `dict` containing the full analysis report:
*   `overallConfidence` (float): 0.0 to 1.0 score of technical quality.
*   `judgement` (str): Human-readable summary (e.g., "Sharp & Well-Exposed").
*   `metrics` (dict): Raw scores for `sharpness`, `exposure`, `noise`.
*   `main_subject_name` (str|None): Detected subject (e.g., "person"), if any.

---

## 🔬 Signal Processing Primitives

Low-level functions for granular analysis. Use these if you are building your own processing pipeline.

### `_calculate_sharpness`

```python
def _calculate_sharpness(gray_image: np.ndarray) -> tuple[float, str]
```
Calculates the FFT Magnitude Spectrum mean.
*   **Returns**: `(score, explanation)`
*   `score`: > 0.7 is typically sharp. < 0.4 is blurry.

### `_calculate_exposure`

```python
def _calculate_exposure(gray_image: np.ndarray) -> tuple[float, dict]
```
Analyzes the luminance histogram using Zone System principles.
*   **Returns**: `(score, zone_data)`
*   `zone_data`: Dict with percentage of pixels in Zone 0 (Black), Zone V (Mid), Zone X (White).

### `_calculate_noise`

```python
def _calculate_noise(gray_image: np.ndarray) -> tuple[float, str]
```
Estimates sensor noise floor using variance in smooth patches.
*   **Returns**: `(score, explanation)`
*   `score`: 1.0 is noiseless. Lower scores mean higher noise.

---

## 📥 RAW Handling

### `_load_image_with_raw_support`

```python
def _load_image_with_raw_support(image_path: str) -> np.ndarray | None
```
Intelligent loader that handles RAW files.
1.  Tries `rawpy` demosaicing (Best Quality).
2.  Fallbacks to `exifread` embedded preview extraction (Fastest).
3.  Fallbacks to standard `cv2.imread`.

**Returns**:
*   `np.ndarray`: BGR image array (standard OpenCV format).
*   `None`: If loading failed.
