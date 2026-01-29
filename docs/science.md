# The Science of Visual Intelligence

This document details the mathematical and neural principles powering the `photo-quality-analyzer` engine. We eschew black-box "AI Magic" in favor of explainable, physics-based signal processing reinforced by modern deep learning.

---

## 1. Sharpness: Fast Fourier Transform (FFT)

**The Problem**: Traditional sharpness metrics (like Laplacian Variance) are sensitive to image noise and content rotation. A noisy, blurry image can score higher than a clean, sharp one.

**The Solution**: We analyze the image in the **Frequency Domain** using the **Fast Fourier Transform (FFT)**.

*   **Logic**: Sharp edges (high frequencies) in the spatial domain correspond to significant energy in the peripheral regions of the magnitude spectrum.
*   **Algorithm**:
    1.  Convert image to Grayscale.
    2.  Compute 2D FFT.
    3.  Shift the zero-frequency component to the center.
    4.  Calculate the magnitude spectrum.
    5.  Compute the average magnitude value.
*   **Why it works**: A blurry image acts as a low-pass filter, suppressing high-frequency energy. The FFT magnitude drops significantly.

**Learn More**:
*   [OpenCV FFT Tutorial](https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html)
*   [Understanding Frequency Domain](https://www.cs.unm.edu/~brayer/vision/fourier.html)

---

## 2. Exposure: The Zone System

**The Problem**: Simple "average brightness" fails because a "correctly" exposed night shot is dark, and a snow scene is bright.

**The Solution**: We implement a digital version of **Ansel Adams' Zone System**.

*   **Logic**: We divide the 8-bit luminance histogram (0-255) into 11 visual zones.
*   **Zones Implemented**:
    *   **Zone 0 (Pure Black, 0-23)**: No detail. Clipping here is bad if accidental.
    *   **Zone V (Middle Gray, 116-139)**: The anchor for "correct" exposure.
    *   **Zone X (Pure White, 232-255)**: Blown highlights. No data recovery possible.
*   **Scoring**: We penalize massive clipping in Zone 0 or Zone X, but allow for high-contrast artistic intent if the dynamic range spans multiple zones.

**Learn More**:
*   [The Zone System Explained](https://luminous-landscape.com/zone-system/)
*   [Ansel Adams' Techniques](https://www.anseladams.com/the-zone-system/)

---

## 3. Noise: Local Variance Profiling

**The Problem**: High ISO creates grain. We need to distinguish between "texture" (detail) and "noise" (sensor artifact).

**The Solution**: We specific use **Local Variance Analysis** on smooth patches.

*   **Algorithm**:
    1.  Divide the image into small patches (e.g., 20x20 pixels).
    2.  Calculate the standard deviation (contrast) of each patch.
    3.  Filter out high-contrast patches (edges/details).
    4.  The remaining low-contrast patches likely represent smooth areas (sky, wall).
    5.  The average variance of these smooth patches is the ESTIMATED NOISE FLOOR.

**Learn More**:
*   [Image Noise Estimation Logic](https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image)

---

## 4. Subject Detection: YOLO12x

**The Problem**: Global metrics fail on shallow depth-of-field portraits. If the background is blurry (bokeh), the whole image scores low on sharpness.

**The Solution**: We use **YOLO12x** (You Only Look Once, v12 Extralarge) for state-of-the-art object detection.

*   **Logic**:
    1.  Run inference to find bounding boxes for `Person`, `Cat`, `Dog`, `Car`.
    2.  **Crop & Score**: We calculate Sharpness specifically *inside* the bounding box.
    3.  **Weighted Average**: The final score is `70% Subject Sharpness + 30% Global Sharpness`.
*   **Context**: If a person is detected, their face/eyes MUST be sharp. The background doesn't matter.

**Learn More**:
*   [Ultralytics YOLO Docs](https://docs.ultralytics.com/)

---

## 5. Composition: Rule of Thirds

**The Problem**: A technically perfect photo can be boring.

**The Solution**: We check geometric alignment of main subjects against a **3x3 Grid**.

*   **Logic**:
    1.  Divide image coordinates into thirds ($x=33\%$, $x=66\%$, etc.).
    2.  Calculate the *Centroid* of the main subject's bounding box.
    3.  Measure Euclidean distance from the nearest "Power Point" (intersection of grid lines).
    4.  Closer alignment = Higher aesthetic score.

**Learn More**:
*   [Rule of Thirds in Photography](https://digital-photography-school.com/rule-of-thirds/)

---

## 6. Color Balance: Neutral Pixel Selection

**The Problem**: Detecting color casts (too warm/cool) without knowing the original scene is hard.

**The Solution**: We assume the "Gray World Hypothesis" applied to **Neutral Pixels**.

*   **Algorithm**:
    1.  Identify pixels that *should* be gray (low saturation, high luminosity).
    2.  Calculate the average R, G, B of these neutral candidates.
    3.  Deviation from `R=G=B` indicates a color cast.
