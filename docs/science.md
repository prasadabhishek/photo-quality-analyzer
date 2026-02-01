# Scientific Foundation of photo-quality-analyzer

This document provides a technical deep-dive into the signal processing and computer vision algorithms used to assess photographic quality.

---

## 1. Sharpness: FFT-Based Acutance
Most sharpness algorithms rely on simple edge gradients (like the Laplacian), which are highly susceptible to sensor noise and rotation.

**Our Approach**: We transform the image into the frequency domain using **Fast Fourier Transform (FFT)**. We analyze the **Anisotropy** and magnitude spectrum moments to determine purely optical clarity.
- **Diffraction Adjustment**: The engine fetches the camera's sensor size and aperture from EXIF. It calculates the **Airy Disk** size to determine if the image is "soft" due to physics (diffraction) rather than poor focus.
- **Reference**: [Cambridge in Colour: Diffraction](https://www.cambridgeincolour.com/tutorials/diffraction-photography.htm)

## 2. Exposure: Ansel Adams Zone System
Instead of a simple mean brightness, we analyze the histogram using the **Zone System**.
- **Zone 0-I**: Pitch black (Blocked shadows).
- **Zone V**: Mid-gray (Ideal exposure).
- **Zone IX-X**: Pure white (Clipping/Blown highlights).
- **Metric**: The exposure score penalizes pixels in Zones 0 and X, while favoring Zone V. Unlike standard "brightness" checks, this allows for high-key or low-key artistic choices as long as "destructive" clipping is avoided.
- **Reference**: [Ansel Adams Zone System](https://en.wikipedia.org/wiki/Zone_System)

## 3. Dynamic Range & Tonal Density: Shannon Entropy
To measure how much information is actually recorded in the tonal range, we use **Shannon Entropy**.
- **Theory**: Tonal range is not just about the distance between black and white; it's about the density of information in between. Entropy measures the "unpredictability" or information density of the pixel intensities.
- **Normalization**: The result is normalized against the camera's **Photons-to-Photos** PDR (Photographic Dynamic Range) baseline.

## 4. Hardware-Aware Normalization
A photo from a Sony A7R V (61MP Full Frame) should not be held to the same absolute acutance standard as an iPhone 15 (12MP). 
- **Database**: We maintain a JSON database of **147+ camera models**.
- **Metrics**: We store pixel pitch, sensor dimensions, and laboratory benchmarked dynamic range for each model.
- **Adjustment**: The engine "grades on a curve" based on the hardware used, ensuring fair comparisons across different gear tiers.

## 5. Subject-Aware Focus: Neural ROI
Traditional "Global Sharpness" fails if the background is intentionally blurred (bokeh).
- **AI Integration**: We use **YOLOv11** to identify subjects (people, animals, cars).
- **ROI Masking**: We calculate sharpness *only* within the bounding box of the detected subject.
- **Focus Fall-off**: We compare subject sharpness to background sharpness to verify that the focus was intentional.
