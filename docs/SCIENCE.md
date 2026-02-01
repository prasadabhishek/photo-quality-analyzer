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

**ELI5**: Imagine looking through a screen door. If the holes are big, you see clearly. If the holes are made very tiny (closing the aperture), the light starts to "bend" around the wires (diffraction) and the image naturally softens. The engine checks camera settings to determine if softness is a result of focus or if it is approaching the physical limits of the lens.

**The Math**:
- We perform a **Fast Fourier Transform (FFT)** to move into the spatial frequency domain.
- We analyze the **Anisotropy Ratio** (Directionality) of the High-Frequency (HF) spectrum using 2nd-order Central Moments.
- **Diffraction Warnings**: The engine calculates the **Diffraction Limited Aperture (DLA)** ($N = \frac{\text{Pixel Pitch}}{1.22 \lambda}$). If the current aperture exceeds this limit (e.g., f/22 on a high-res sensor), the engine reports the raw sharpness score but appends a **Technical Warning** ("Diffraction Limit Reached"). This ensures honest reporting of physical softness rather than artificially boosting the score.

---

### B. Exposure: Ansel Adams Zone System

**ELI5**: Think of a photo like a coloring book. If a spot is colored pure white (blown highlights) or pure black (crushed shadows), the original drawing is lost. The engine evaluates the image to ensure most details are in the "middle" where they are clearly visible.

**The Logic**:
- The histogram is divided into 11 zones (0-X) based on the Zone System.
- **Zone 0-I**: Destructive "Crushed" shadows.
- **Zone V**: Ideal 18% gray (Middle Gray).
- **Zone IX-X**: Destructive "Blown" highlights.
- **Zone IX-X**: Destructive "Blown" highlights.
- **Subject-Aware Metering**: If a subject is detected (via YOLO), the engine calculates the average luminance of the **Subject's Bounding Box** and targets Zone V (18% gray) for that specific region. This ensures that backlit portraits are scored correctly even if the background is blown out.
- **Global Fallback**: If no subject is found, the engine evaluates the global histogram, penalizing clipping in Zones 0 and X.
- **Shutter-Awareness**: At fast shutter speeds (action shots), the engine grants higher tolerance for highlight clipping to prioritize frozen motion.

---

### C. Noise: ISO-Adaptive Variance Sampling

**ELI5**: Imagine listening to music with some background static. If you are in a quiet room (Low ISO), static is very noticeable. If you are at a loud concert (High ISO), a little bit of static is expected and less distracting. The engine adjusts its expectations based on how "loud" the sensor was set to.

**The Process**:
- The image is divided into an 8x8 grid of patches.
- We calculate the variance ($\sigma^2$) for each patch.
- **Chroma vs. Luma**: The engine separates noise into two components using the **LAB Color Space**:
    1.  **Luminance (L channel)**: Treated as "Grain". Normalized generously to allow for filmic texture.
    2.  **Chrominance (A/B channels)**: Treated as "Digital Noise". Penalized heavily (0.4 weight) as color blotches are rarely desirable.
- **ISO-Adaptive**: The noise floor is dynamically normalized based on the **ISO setting**. A clean image at ISO 12,800 is rated relative to the expected photon shot noise at that gain level.

---

### D. Dynamic Range: Tonal Entropy

**ELI5**: Think of a box of 256 crayons. If a photo only uses 5 shades of gray, it looks "flat." If it uses a wide variety of "crayons" from the brightest white to the darkest shadow, it has "high dynamic range." The engine counts how much of that variety is present in the image.

**The Metric**:
- **Previous Flaw**: Shannon Entropy ($H$) conflated **Noise** with **Detail**. A noisy gray card would score higher than a clean gradient.
- **New Metric**: **98th-Percentile Histogram Width**. 
- **The Math**: We calculate the width of the histogram that contains the central 98% of pixel values. This ignores salt-and-pepper noise outliers and measures the actual **Tonal Utilization** of the sensor's bit depth.
- **Benchmarking**: The result is normalized against our internal database of **Photons-to-Photos PDR** curves, ensuring results are comparable across different sensor sizes.

---

## 4. Visual Intelligence & Neural ROI

### YOLOv11 Object Detection

**ELI5**: If you take a picture of a dog, the dog should be sharp, but it's often okay (or even preferred) if the trees behind it are blurry. The engine identifies the main subject so it can judge the focus where it matters most.

1.  **ROI Masking**: Instead of grading global sharpness, the engine prioritizes the bounding box of the main subject.
2.  **Intent Check**: If the subject is sharp but the background has "bokeh" (intentional blur), the engine recognizes this as a stylistic choice rather than a technical failure.

### Composition: Headroom Analysis
The engine uses psychophysical heuristics to evaluate framing for portraits:

1.  **Rule of Thirds**: Distances centroids to Power Points.
2.  **Headroom Analysis**: Calculates the vertical space above the subject's head box.
    - **< 2%**: Penalty (Chopped head / Claustrophobic).
    - **> 35%**: Penalty (Excessive dead space).
    - **Ideal Range**: 8-20% of frame height.

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
