# Releasing photo-quality-analyzer-core

This guide details the process for publishing the `photo-quality-analyzer-core` SDK to PyPI.

## 📋 Prerequisites
Ensure you have the build tools installed:
```bash
pip install build twine
```

## 🚀 1. Bump Version
Update the `version` string in `pyproject.toml`:
```toml
[project]
version = "0.2.0"  # <--- Update this
```

## 📦 2. Build Artifacts
Clean previous builds and generate new distributions:
```bash
rm -rf dist/
python -m build
```
*Output should be in `dist/`: `photo_quality_analyzer_core-0.2.0.tar.gz` and `photo_quality_analyzer_core-0.2.0-py3-none-any.whl`.*

## 🧪 3. Verify
Check the package metadata:
```bash
twine check dist/*
```

## ☁️ 4. Publish (TestPyPI First)
Start by uploading to TestPyPI:
```bash
twine upload --repository testpypi dist/*
```
*Verify on [test.pypi.org](https://test.pypi.org/project/photo-quality-analyzer-core/).*

## 🌍 5. Publish (Production)
Upload to the real PyPI:
```bash
twine upload dist/*
```
*Verify on [pypi.org](https://pypi.org/project/photo-quality-analyzer-core/).*

## 🏷️ 6. Git Tag
Tag the release in git:
```bash
git tag v0.2.0
git push origin v0.2.0
```
