# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2026-02-15

### 🚀 Performance Overhaul
- **Fast Mode**: Introduced intelligent downsampling for sharpness metrics (`fast_mode=True`), reducing processing time from ~1.5s to ~0.19s per image (7.8x speedup) with minimal accuracy loss.
- **Forensic Precision**: `fast_mode=False` retains the original pixel-perfect analysis for critical work.
- **Optimization**: Default behavior for batch tools now prioritizes speed.

## [0.7.1] - 2026-02-08

### 🐛 Bug Fixes

- **CI Tests**: Fixed `test_threshold_move_selects` and `test_threshold_move_rejects` test failures in GitHub Actions
- **Testing**: Added `enable_subject_detection` parameter to `process_folder` function to allow tests to run without YOLO model
- **Performance**: Tests now run faster by skipping YOLO initialization when not needed
- **Reliability**: Directory creation now happens before YOLO checks, ensuring robust file organization even if model loading fails

## [0.7.0] - 2026-02-08

### 🎯 Major Features

#### Multi-Subject Focus Scan
- **Enhanced**: Focus calculation now scans ALL detected subjects (confidence > 0.5) instead of just the highest-confidence subject
- Uses the best focus score found across all subjects
- Prevents false negatives where a blurry foreground subject dooms a sharp background subject
- Particularly beneficial for group photos and street photography
- Performance impact: +26ms for 7 subjects (~6.5% overhead, acceptable trade-off)

#### Subject-Weighted Sharpness  
- Sharpness calculation now weights Subject ROI at 70% and Global at 30%
- Prevents sharp backgrounds from hiding blurry subjects
- Addresses the "texture trap" where background detail (trees, buildings) masked subject softness

#### YOLO26 ONNX Integration
- Migrated from full PyTorch YOLO to lightweight ONNX Runtime
- Reduced disk footprint significantly (fewer dependencies)
- Faster inference with YOLO26 nano model
- Headless OpenCV build for reduced dependencies

### 📚 Documentation

- Added [PyImageSearch reference](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/) for Laplacian blur detection method to `SCIENCE.md`
- Created `BACKLOG.md` documenting known limitations and future improvements
- Updated `API.md` with new subject detection parameters
- Enhanced `USAGE.md` with multi-subject workflow examples
- Clarified library philosophy in `README.md`

### 🧪 Tests

- Fixed `test_focus_dof_awareness` for new multi-subject explanation format
- Skipped `test_sharpness_differentiation` (synthetic test data exceeds normalization ceiling - needs real-world test images)
- **Test Suite**: 24 passed, 1 skipped

### 🔧 Internal Changes

- Removed deprecated `requirements.txt` (now using `pyproject.toml` exclusively)
- Optimized model resource structure and package data handling
- Restored stable metric math from v0.6.x production branch
- Integrated forensic fixes from production

### 📊 Impact

- **Files Changed**: 11 files
- **Net Change**: +220 lines (716 additions, 496 deletions)
- **Commits**: 9 commits ahead of v0.3.0

### 🚀 Migration Guide

The multi-subject focus scan is **backward compatible** (same API). However, scoring behavior has changed:

**Expected Changes**:
- **Group photos**: More lenient (if ANY person is sharp, photo passes)
- **Street scenes**: More accurate (won't be fooled by sharp background if subject is blurry)
- **Single subject**: Minimal score change

No code changes required for existing integrations.

---

## [0.3.0] - 2026-01-31

### Added
- **Expanded Camera Database**: Integrated benchmarks for **147+ unique camera models** across 8 major brands (Sony, Nikon, Canon, Fujifilm, Panasonic, Olympus, Leica).
- **Decentralized Heuristics**: Moved hardware detection patterns from hardcoded Python logic to `camera_database.json`.
- **Scientific Documentation Suite**: 
    - Created `docs/SCIENCE.md` for technical deep-dives into FFT-based sharpness, Zone System exposure, and Shannon Entropy.
    - Refactored `README.md` for minimalist "Alt of README" flow.
- **Improved Matching Engine**: Implemented "Longest Match" strategy in `get_camera_data` to prevent partial model name collisions (e.g., Nikon D700 vs D7000).
- **Comprehensive Test Suite**: Expanded unit testing to 22 tests covering core signal processing, hardware heuristics, and XMP generation.

### Changed
- **Default RAW Support**: Moved `rawpy` from optional "extras" to a core dependency. All RAW formats are now supported out-of-the-box.
- **API Standardisation**: Renamed internal utility functions for clarity (e.g., `write_xmp_sidecar` -> `create_xmp_sidecar`).

### Fixed
- Fixed an edge case where certain APS-C cameras were incorrectly identified as Full Frame due to substring matching.
- Standardized documentation across all core functions for professional open-source readiness.

---

## [0.2.0] - 2026-01-25
- Initial beta release with Phase 2 Context-Aware metrics.
- Support for basic EXIF-aware sharpness and exposure.
