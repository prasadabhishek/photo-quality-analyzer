# Scientific Documentation: photo-quality-analyzer

This document provides a comprehensive technical deep-dive into the signal processing, computer vision, and optical physics foundations of the **Photo Quality Analyzer**.

---

## 1. High-Level Workflow

The engine follows a structured pipeline to transform raw pixel data into human-readable photographic judgements:

1.  **Ingestion**: Format detection and high-fidelity loading (RAW/JPEG).
2.  **Context Extraction**: EXIF parsing for hardware metadata (Aperture, ISO, Model).
3.  **Neural Detection**: YOLOv11 subject identification and ROI definition.
4.  **Signal Analysis**: Parallel computation of frequency, tonal, and statistical metrics.
5.  **Normalization**: Benchmarking results against a database of **147+ camera models**.
6.  **Synthesis**: Weighted averaging and linguistic mapping to final labels.

---

## 2. Ingestion & RAW Pipeline

For professional photographers, the ability to analyze RAW files directly is critical. The engine implements a 3-tier loading strategy:

1.  **RAW De-mosaicing ([LibRaw](https://www.libraw.org/))**: Uses the `rawpy` wrapper to extract 16-bit linear signal data. The engine uses a "Turbo Optimization" (`half_size=True`) to ensure sub-second analysis of 60MP+ files.
2.  **High-Res Preview Recovery**: If de-mosaicing fails, it parses the EXIFMakerNote for the largest embedded JPEG preview (often full-resolution).
3.  **Standard Decoding**: Falls back to `OpenCV`'s hardware-accelerated decoders.

---

## 3. Core Technical Metrics

### A. Sharpness: FFT Anisotropy & Diffraction
Standard sharpness checks (like Laplacian Variance) are easily fooled by noise or directional texture.

**The Math**:
- We perform a **Fast Fourier Transform (FFT)** to move into the spatial frequency domain.
- We analyze the **Anisotropy Ratio** (Directionality) of the High-Frequency (HF) spectrum using 2nd-order Central Moments.
- **Aperture-Awareness**: The engine fetches the camera's sensor size and aperture. It calculates the **Airy Disk** diameter ($D = 2.44 \cdot \lambda \cdot N$). If the aperture ($N$) is beyond the **Diffraction Limited Aperture (DLA)** of the sensor, the sharpness score is normalized to reflect the physical limits of the glass, rather than user error.

### B. Exposure: Ansel Adams Zone System
The engine moves beyond simple "mean brightness" by applying the **Zone System** developed by Ansel Adams.

**The Logic**:
- The histogram is divided into 11 zones (0-X).
- **Zone 0-I**: Destructive "Crushed" shadows.
- **Zone V**: Ideal 18% gray (Middle Gray).
- **Zone IX-X**: Destructive "Blown" highlights.
- **Metric**: The score calculates the deviance from Zone V while applying heavy nonlinear penalties for clipping in Zone 0 or X.
- **Shutter-Awareness**: At fast shutter speeds (action shots), the engine grants higher tolerance for highlight clipping to favor frozen motion.

### C. Noise: ISO-Adaptive Variance Sampling
Noise is estimated by sampling statistical variance in low-texture regions of the frame.

**The Process**:
- The image is divided into an 8x8 grid.
- We calculate the variance ($\sigma^2$) for each patch.
- We identify the 5 "smoothest" patches (lowest variance) to isolate the sensor's **Noise Floor** from actual image detail.
- **Normalization**: The noise score is dynamically scaled based on the **ISO setting**. A clean image at ISO 12,800 is rated significantly higher than an equally clean image at ISO 100.

### D. Dynamic Range: Tonal Entropy
Dynamic Range is measured via **Shannon Entropy**, which treats the tonal distribution as an information channel.

**The Metric**:
- **Formula**: $H = -\sum P(x) \log_2 P(x)$
- Max entropy ($H=8.0$) represents a perfectly distributed 8-bit tonal range.
- **Benchmarking**: The result is normalized against our internal database of **Photons-to-Photos PDR** curves, ensuring a smartphone isn't unfairly compared to a Medium Format sensor.

---

## 4. Visual Intelligence & Neural ROI

### YOLOv11 Object Detection
The engine uses a neural network to understand *what* is in the frame. This is critical for **Subject-Aware Sharpness**.

1.  **ROI Masking**: Instead of grading global sharpness, the engine prioritizes the bounding box of the main subject (e.g., a person or animal).
2.  **Intent Check**: If the subject is sharp but the background has "bokeh" (intentional blur), the engine rewards the photo for technical mastery rather than penalizing it for background softness.

### Composition: Rule of Thirds
The engine calculates the Euclidean distance between the centroids of detected subjects and the four "Power Points" of the Rule of Thirds grid. 

---

## 5. The Synthesis Engine

The final `overallConfidence` is calculated using a weighted gatekeeper formula:

$$Score = Tech \cdot (0.8 + 0.2 \cdot Aesthetic)$$

**Weights**:
- **Technical (60%)**: Sharpness (40%), Focus (30%), Exposure (20%), Noise (10%).
- **Aesthetic (40%)**: Dynamic Range (40%), Color Balance (40%), Composition (20%).

**Linguistic Mapping**:
Final scores are mapped to a qualitative scale used in XMP sidecars and CLI output:
- **Excellent**: ≥ 0.8 (Award 5 Stars)
- **Good**: ≥ 0.65 (Award 3 Stars)
- **Acceptable**: ≥ 0.5 (Keep)
- **Poor**: < 0.35 (Rejected Label)

---

## 6. Resources & References
- **Optical Theory**: [Cambridge in Colour](https://www.cambridgeincolour.com/)
- **Sensor Benchmarks**: [DXOMARK](https://www.dxomark.com/)
- **Dynamic Range Curves**: [PhotonsToPhotos](https://www.photonstophotos.net/)
- **XMP Standard**: [Adobe XMP Core Specification](https://www.adobe.com/products/xmp.html)
