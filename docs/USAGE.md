# Usage Guide: photo-quality-analyzer

This guide provides practical examples for integrating the **Photo Quality Analyzer** into your workflows, whether you are using the CLI or building custom Python applications.

---

## 1. Quick Start (Python API)

The simplest way to use the library is the `evaluate_photo_quality` function.

```python
from photo_quality_analyzer_core.analyzer import evaluate_photo_quality

# Analyze a single image
result = evaluate_photo_quality("my_photo.jpg")

# Print the final verdict
print(f"Judgement: {result['judgement']}")
print(f"Confidence: {result['overallConfidence']:.2f}")
print(f"AI Description: {result['description']}")
```

---

## 2. Advanced Python Usage

### A. Performance Optimization (Disabling AI)
Object detection (YOLO) adds ~1-2s of latency. If you only need physics-based metrics (Sharpness/Exposure), you can disable it:

```python
# Ultra-fast mode: Only runs signal processing (~0.05s)
result = evaluate_photo_quality(
    "photo.jpg", 
    enable_subject_detection=False 
)
```

### B. Requesting Specific Metrics
If you only care about specific aspects (e.g., just Sharpness and Noise), you can save processing time:

```python
result = evaluate_photo_quality(
    "photo.jpg",
    requested_metrics=["sharpness", "noise"]
)

# Only these keys will be present in result['metrics']
print(result['metrics']['sharpness']['score'])
```

### C. RAW File Processing
The library automatically detects RAW files. High-fidelity analysis requires `rawpy`:

```bash
pip install "photo-quality-analyzer-core[raw]"
```

```python
# The same function handles .ARW, .CR2, .NEF, etc.
result = evaluate_photo_quality("shot.ARW")
```

---

## 3. Command Line Interface (CLI)

The package includes a powerful CLI for bulk analysis and organization.

### Basic Batch Analysis
Scan a folder and see results in the terminal:
```bash
photographi --folder_path ./my_shoot
```

### Automatic Sorting
Move files into `good/`, `fair/`, and `bad/` folders based on AI judgement:
```bash
photographi --folder_path ./my_shoot --move
```

### Detailed Output
See the full JSON breakdown for every single photo:
```bash
photographi --folder_path ./my_shoot --verbose
```

---

## 4. Understanding the JSON Output

When you call `evaluate_photo_quality`, you get a structured dictionary:

```json
{
  "overallConfidence": 0.88,
  "judgement": "Excellent",
  "description": "Image containing: person, dog, mountain.",
  "metrics": {
    "sharpness": {
      "score": 0.92,
      "explanation": "Edges are sharp and directional."
    },
    "exposure": {
      "score": 0.75,
      "explanation": "Exposure is technically sound."
    }
  },
  "metadataStatus": "present"
}
```

- **`overallConfidence`**: A weighted average (0.0 - 1.0) of technical and aesthetic factors.
- **`judgement`**: A human-friendly label used for sorting.
- **`description`**: A list of objects identified by the AI in the scene.

---

## 5. Integration with Lightroom/Capture One

The library can generate **Adobe-compatible XMP sidecars**. This allows you to "cull" your photos with AI and then see the ratings immediately when you open them in Lightroom.

```python
from photo_quality_analyzer_core.analyzer import evaluate_photo_quality, create_xmp_sidecar

image = "landscape.jpg"
result = evaluate_photo_quality(image)

# This creates 'landscape.xmp' with a Star Rating based on quality
create_xmp_sidecar(image, result['judgement'], result['overallConfidence'])
```

## 6. Customizing Camera Database (v0.4.0+)
If your camera is not in the bundled database, you can provide your own specifications without waiting for an update.

### A. Environment Variable
Point to a JSON file containing your camera data:
```bash
export PQA_CAMERA_DB_PATH="/path/to/my_cameras.json"
```

### B. User Config
Place a file at `~/.photo_quality_analyzer/camera_database.json`. The library will automatically load and merge it with the built-in data.

**JSON Format**:
```json
{
    "Sony": {
        "A7RV": {
            "dr": 14.8,
            "sensor": "full_frame",
            "aliases": ["ILCE-7RM5"]
        }
    }
}
```

## 7. YOLO-NAS Engine (v0.5.0+) - Advanced Users

### What is YOLO-NAS?
YOLO-NAS uses Neural Architecture Search for potentially better accuracy and speed on certain hardware configurations.

### Why is this Optional?

> **TL;DR**: The default YOLO engine (v8/v11) works great for 99% of users. YOLO-NAS is only beneficial for specific edge devices or research purposes.

YOLO-NAS requires the `super-gradients` library (~500-800 MB with dependencies) which has several compatibility limitations:

- **Outdated Dependencies**: Pins old versions of `onnxruntime`, `torchmetrics`, and other packages that conflict with modern ML environments
- **Platform-Specific**: Installation fails on many systems due to architecture-specific builds (ARM Macs, newer Python versions, etc.)
- **Limited Benefit**: For general photography analysis, the default YOLO models provide equivalent or better results

**When to use YOLO-NAS:**
- You're deploying to NVIDIA Jetson or specific edge hardware
- You're already invested in Deci.ai's ecosystem
- You're conducting model benchmarking research

### Installation (If Supported)

```bash
pip install super-gradients
```

If installation fails with dependency errors, your platform is not compatible. Continue using the default `--engine yolo` (no flag needed).

### Usage
```bash
# Only if super-gradients installed successfully
photographi --engine yolo-nas --model_path yolo_nas_s.pt --folder_path ./photos
```
