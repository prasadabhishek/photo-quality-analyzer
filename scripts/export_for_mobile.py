"""
Export YOLO model and config for mobile (Core ML + TFLite) in a privacy-first, offline workflow.
- Reads config.ini and mirrors values to mobile_config.json
- Extracts label names from the Ultralytics YOLO model
- Exports TFLite (with optional quantization) and Core ML

Run:
  python3 scripts/export_for_mobile.py --model ./yolo12x.pt --config ./config.ini --outdir ./mobile_artifacts --quant fp16

Notes:
- Keep operations offline by using a local model path.
- Ensure ultralytics is installed in your Python venv.
"""
from __future__ import annotations
import os, json, configparser, argparse
from typing import Any, Dict, List

try:
    from ultralytics import YOLO
except Exception as e:
    # Allow running in --skip_model mode without ultralytics installed
    YOLO = None  # type: ignore


def _read_config(cfg_path: str) -> Dict[str, Any]:
    cfg = configparser.ConfigParser()
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg.read(cfg_path)

    def gf(section: str, key: str, fallback=None, cast=float):
        try:
            if cast is float: return cfg.getfloat(section, key, fallback=fallback)
            if cast is int: return cfg.getint(section, key, fallback=fallback)
            return cfg.get(section, key, fallback=fallback)
        except Exception:
            return fallback

    return {
        "NormalizationFactors": {
            "sharpness": gf("NormalizationFactors", "sharpness", 1000.0),
            "focus_area": gf("NormalizationFactors", "focus_area", 1000.0),
            "noise": gf("NormalizationFactors", "noise", 50.0),
        },
        "Thresholds": {
            "exposure_ideal_mean": gf("Thresholds", "exposure_ideal_mean", 128.0),
            "dynamic_range_max": gf("Thresholds", "dynamic_range_max", 255.0),
            "yolo_confidence": gf("Thresholds", "yolo_confidence", 0.5),
            "yolo_nms": gf("Thresholds", "yolo_nms", 0.45),
        },
        "Weights": {
            "overall_tech": gf("Weights", "overall_tech", 0.6),
            "overall_other": gf("Weights", "overall_other", 0.4),
        },
        "Models": {
            "default_yolo_model": gf("Models", "default_yolo_model", "yolo12x.pt", cast=str),
        },
        "JudgementLevels": {
            "excellent": gf("JudgementLevels", "excellent", 0.9),
            "good": gf("JudgementLevels", "good", 0.7),
            "fair": gf("JudgementLevels", "fair", 0.5),
            "poor": gf("JudgementLevels", "poor", 0.3),
        },
    }


def _save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load_labels_from_file(path: str) -> List[str]:
    labels: List[str] = []
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            if name:
                labels.append(name)
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo12x.pt")
    ap.add_argument("--config", default="config.ini")
    ap.add_argument("--outdir", default="mobile_artifacts")
    ap.add_argument("--quant", choices=["int8", "fp16", "fp32"], default="fp16")
    ap.add_argument("--skip_model", action="store_true", help="Skip loading/exporting the model; only write JSON artifacts.")
    ap.add_argument("--labels_path", help="Optional path to a newline-delimited labels file to build coco_labels.json when skipping model.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Config mirror JSON (always write)
    cfg = _read_config(args.config)
    _save_json(cfg, os.path.join(args.outdir, "mobile_config.json"))

    labels: List[str] = []

    if args.skip_model:
        # Optional labels file support
        if args.labels_path and os.path.exists(args.labels_path):
            labels = _load_labels_from_file(args.labels_path)
        _save_json(labels, os.path.join(args.outdir, "coco_labels.json"))
        print("Skipped model export by request (--skip_model). Wrote config and labels JSON.")
        return

    # Below requires ultralytics and a local .pt file
    if YOLO is None:
        raise SystemExit("Ultralytics not available. Install with `pip install ultralytics` or use --skip_model.")

    model_path = args.model
    if not os.path.exists(model_path):
        raise SystemExit(f"Model file not found: {model_path}. Provide a local .pt or use --skip_model.")
    if os.path.isdir(model_path):
        raise SystemExit(f"Model path points to a directory, please provide a .pt file: {model_path}")

    model = YOLO(model_path)

    # Labels from model
    if hasattr(model, "names") and isinstance(model.names, dict) and model.names:
        max_id = max(model.names.keys())
        labels = [f"unknown_id_{i}" for i in range(max_id + 1)]
        for k, v in model.names.items():
            labels[int(k)] = str(v)
    _save_json(labels, os.path.join(args.outdir, "coco_labels.json"))

    # Export TFLite
    tflite_args: Dict[str, Any] = {"format": "tflite"}
    if args.quant == "int8":
        tflite_args.update({"int8": True})
    elif args.quant == "fp16":
        tflite_args.update({"half": True})
    tflite_path = model.export(**tflite_args)

    # Export Core ML
    coreml_path = model.export(format="coreml")

    print("Artifacts generated in:", os.path.abspath(args.outdir))
    print("Labels:", os.path.abspath(os.path.join(args.outdir, "coco_labels.json")))
    print("Config:", os.path.abspath(os.path.join(args.outdir, "mobile_config.json")))
    print("TFLite:", tflite_path)
    print("CoreML:", coreml_path)


if __name__ == "__main__":
    main()
