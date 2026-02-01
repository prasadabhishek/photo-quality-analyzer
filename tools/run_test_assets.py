
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from photo_quality_analyzer_core.analyzer import evaluate_photo_quality, ensure_yolo_initialized

def main():
    # Assets are in the parent photographi workspace
    assets_dir = "/Users/abhishekprasad/workspace/photographi/local_test_assets"
    if not os.path.exists(assets_dir):
        print(f"Error: {assets_dir} not found")
        return

    # Pre-load YOLO
    ensure_yolo_initialized(model_size='n')

    files = sorted([f for f in os.listdir(assets_dir) if f.lower().endswith(('.jpg', '.arw'))])
    
    print(f"Found {len(files)} files in {assets_dir}")
    
    for f in files:
        path = os.path.join(assets_dir, f)
        print(f"\n--- Analyzing {f} ---")
        try:
            # Request all scientific metrics
            metrics = ["sharpness", "exposure", "focus", "noise", "dynamicRange", "composition", "color"]
            result = evaluate_photo_quality(path, requested_metrics=metrics, enable_subject_detection=True)
            
            # Print key results concisely
            print(f"Judgement: {result.get('judgement')}")
            print(f"Confidence: {result.get('overallConfidence', 0):.4f}")
            
            # Check detailed metrics if available in return (it returns a flat dict mostly, but let's see)
            # Actually evaluate_photo_quality returns a comprehensive dict. 
            # I'll print the 'metrics' or specific scores if accessible, or just the whole thing pretty-printed.
            print(json.dumps(result, indent=2, default=str))
            
        except Exception as e:
            print(f"Failed to analyze {f}: {e}")

if __name__ == "__main__":
    main()
