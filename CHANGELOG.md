# Changelog

All notable changes to this project will be documented in this file.

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
