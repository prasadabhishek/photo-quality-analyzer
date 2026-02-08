# Future Enhancements Backlog

This document tracks potential improvements to the Photo Quality Analyzer based on scientific research and user feedback.

---

## Phase 1: Low-Hanging Fruit

### 1.1 Content Saliency Check
**Problem**: A sharp photo of a blank wall scores 0.8 (Excellent) due to high technical scores.

**Solution**: Add a "visual interest" metric that penalizes images with low aesthetic scores more severely.

**Implementation**:
- Detect featureless regions (e.g., >70% uniform color)
- Apply penalty multiplier to overall score if content saliency is low
- Threshold: If `aesthetic_score < 0.3` and `feature_density < 0.2`, apply `0.5x` technical multiplier

**Complexity**: Low  
**Impact**: Medium

---

### 1.2 Histogram Uniformity (Flatness Metric)
**Problem**: Silhouette images (bimodal histograms) score high on "dynamic range" despite having zero mid-tone detail.

**Solution**: Measure histogram distribution uniformity, not just width.

**Implementation**:
- Calculate histogram entropy or chi-squared test against uniform distribution
- Penalize histograms with >80% of pixels concentrated in <20% of bins
- Add as a sub-component to the Dynamic Range score

**Complexity**: Low  
**Impact**: Low-Medium

---

## Phase 2: Medium Complexity

### 2.1 Adaptive Skin Tone Exposure Targeting
**Problem**: Zone V (18% gray) assumption causes bias. Light skin is underexposed, dark skin is overexposed.

**Solution**: Estimate skin tone reflectance and adjust target zone dynamically.

**Implementation**:
- Use YOLO class `person` + color histogram in bounding box to estimate skin tone
- Map to Monk Skin Tone Scale (or similar) → 10 categories
- Set adaptive target:
  - Light skin (Categories 1-3): Target Zone VI (36% gray)
  - Medium skin (Categories 4-7): Target Zone V (18% gray)
  - Dark skin (Categories 8-10): Target Zone IV (12% gray)

**Complexity**: Medium  
**Impact**: High (addresses documented photographic bias)

**References**:
- [Monk Skin Tone Scale](https://skintone.google/)
- "The Shirley Card and the Color of Film" (Lorna Roth, 2009)

---

### 2.2 Frequency-Based Noise Differentiation (DCT)
**Problem**: Variance-based noise estimation confuses bokeh (smooth gradients) with low noise.

**Solution**: Use Discrete Cosine Transform (DCT) to analyze frequency signature of smooth patches.

**Implementation**:
- Replace or augment 8x8 variance with DCT coefficient analysis
- Noise has "white" spectral signature (high-frequency energy)
- Bokeh gradients have low-frequency energy
- Calculate ratio: `high_freq_energy / low_freq_energy` to distinguish noise from texture

**Complexity**: Medium  
**Impact**: Medium

---

## Phase 3: Research-Level

### 3.1 Natural Scene Statistics (NSS) for Sharpness
**Problem**: FFT anisotropy fails on naturally directional textures (brick walls, fences).

**Solution**: Model the frequency decay pattern to differentiate motion blur from sharp geometric subjects.

**Implementation**:
- Compute radial frequency decay in FFT domain
- Motion blur: exponential decay in perpendicular direction
- Sharp texture: uniform energy in all orthogonal directions
- Train a simple classifier to distinguish the two patterns

**Complexity**: High  
**Impact**: Low (edge case mitigation)

**References**:
- "Blind Image Deblurring Using Dark Channel Prior" (Pan et al., 2016)
- "No-Reference Image Quality Assessment Based on Natural Scene Statistics" (Mittal et al., 2012)

---

### 3.2 Learned "Interestingness" Model
**Problem**: The library cannot judge artistic merit or emotional impact.

**Solution**: Train a lightweight model on curated datasets to predict "visual interest."

**Implementation**:
- Use a pre-trained vision model (e.g., CLIP embeddings)
- Fine-tune on datasets like AVA (Aesthetic Visual Analysis) or IAD (Image Aesthetic Dataset)
- Output: "Interestingness Score" (0-1) as a separate metric
- Do NOT blend with technical scores; keep them separate

**Complexity**: Very High  
**Impact**: High (moves toward "Curator" functionality)

**References**:
- AVA Dataset: [Aesthetic Visual Analysis](http://refbase.cvc.uab.es/files/MMP2012a.pdf)
- NIMA: [Neural Image Assessment](https://arxiv.org/abs/1709.05424)

---

## Non-Goals

The following are **explicitly out of scope** for this library:

- **Subjective Artistic Assessment**: We are not building a "taste engine."
- **Genre-Specific Rules**: No special handling for "street photography" vs. "portraits" vs. "landscapes."
- **Emotion Recognition**: We do not attempt to judge "mood" or "storytelling."

This library remains a **Technical Quality Filter**, not a creative advisor.
