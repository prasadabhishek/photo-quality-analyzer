import unittest
import numpy as np
import os
import sys

# Add the paths to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../photo-quality-analyzer')))

from photo_quality_analyzer_core.analyzer import (
    _calculate_sharpness, 
    _calculate_exposure, 
    _calculate_dynamic_range, 
    _calculate_focus_area,
    SENSOR_SIZES,
    CAMERA_DYNAMIC_RANGE
)
import photo_quality_analyzer_core.analyzer as analyzer

class TestPhase2Metrics(unittest.TestCase):
    """Test suite for Phase 2 Context-Aware Metrics."""

    def setUp(self):
        # Create a clean image (100x100 gray)
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.gray = np.zeros((100, 100), dtype=np.uint8)
        # Add high-frequency texture but ULTRA LOW intensity to avoid 1.0 ceiling
        # Checkerboard pattern in top-left (Mock ROI)
        self.gray[0:2, 0:2] = 1 # Only 4 pixels for focus
        # Texture for sharpness - just a few pixels
        self.gray[50:52, 50:52] = 1
        # Sync RGB
        for c in range(3):
            self.img[:, :, c] = self.gray






    def test_sharpness_aperture_adjustment(self):
        """Test that sharpness score is adjusted based on aperture."""
        # 1. Base case: f/8 (optimal)
        metadata_f8 = {"aperture": 8.0}
        score_f8, expl_f8 = _calculate_sharpness(self.gray, metadata_f8)
        
        # 2. Diffraction-limited: f/22
        metadata_f22 = {"aperture": 22.0}
        score_f22, expl_f22 = _calculate_sharpness(self.gray, metadata_f22)
        
        print(f"\nSharpness context test:")
        print(f"f/8 score: {score_f8:.4f}, Explanation: {expl_f8}")
        print(f"f/22 score: {score_f22:.4f}, Explanation: {expl_f22}")
        
        self.assertIn("Diffraction-limited", expl_f22)
        # At same raw sharpness, f/22 should score HIGHER because it's adjusted for physics
        self.assertGreater(score_f22, score_f8)

    def test_exposure_action_tolerance(self):
        """Test that exposure scoring is more lenient for fast shutter speeds."""
        # Create a slightly clipped image (4% highlight clipping)
        # highlight_tolerance is 0.02 for general, 0.05 for action
        clipped_gray = np.zeros((100, 100), dtype=np.uint8) + 128
        clipped_gray[:20, :20] = 255 # 400 pixels = 4% (0.04)
        
        # 1. Slow shutter speed (1/50s) -> Tolerance 0.02 -> Penalty applies
        metadata_slow = {"shutter_speed": 0.02}
        score_slow, expl_slow = _calculate_exposure(clipped_gray, metadata_slow)
        
        # 2. Fast shutter speed (1/1000s) -> Tolerance 0.05 -> Penalty is 0
        metadata_fast = {"shutter_speed": 0.001}
        score_fast, expl_fast = _calculate_exposure(clipped_gray, metadata_fast)
        
        print(f"\nExposure context test:")
        print(f"1/50s score: {score_slow:.4f}, Explanation: {expl_slow}")
        print(f"1/1000s score: {score_fast:.4f}, Explanation: {expl_fast}")
        
        # Fast shutter should be more lenient
        self.assertGreater(score_fast, score_slow)
        self.assertIn("action", expl_fast.lower())

    def test_dynamic_range_camera_baseline(self):
        """Test that DR scoring accounts for camera capabilities."""
        # 1. Low-end camera
        metadata_low = {"camera_model": "Sony RX100 VII"} # 11 stops
        score_low, _ = _calculate_dynamic_range(self.gray, metadata_low)
        
        # 2. High-end camera
        metadata_high = {"camera_model": "Sony A7R V"} # 14.8 stops
        score_high, _ = _calculate_dynamic_range(self.gray, metadata_high)
        
        print(f"\nDR context test:")
        print(f"RX100 score: {score_low:.4f}")
        print(f"A7RV score: {score_high:.4f}")
        
        # Lower capability camera gets a HIGHER score for the same content 
        # because it's utilizing more of its available potential.
        self.assertGreater(score_low, score_high)

    def test_focus_dof_awareness(self):
        """Test that focus scoring is more lenient for shallow DOF."""
        # Mock YOLO model presence to enter the logic block
        original_model = analyzer.g_yolo_model
        
        class MockBoxList:
            def __init__(self, box):
                self.box_obj = box
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return self.box_obj
            @property
            def xyxy(self):
                return type('obj', (object,), {'cpu': lambda s: type('obj', (object,), {'numpy': lambda s: np.array([[0, 0, 10, 10]])})()})()
            @property
            def conf(self):
                return type('obj', (object,), {'cpu': lambda s: type('obj', (object,), {'numpy': lambda s: np.array([0.9])})()})()
            @property
            def cls(self):
                return type('obj', (object,), {'cpu': lambda s: type('obj', (object,), {'numpy': lambda s: np.array([0])})()})()

        class MockModel:
            def __call__(self, *args, **kwargs):
                return self.predict(*args, **kwargs)
            def predict(self, *args, **kwargs):
                class MockResult:
                    def __init__(self):
                        self.boxes = MockBoxList(None)
                return [MockResult()]

        analyzer.g_yolo_model = MockModel()
        analyzer.g_coco_names = ["person"]
        
        try:
            base_score = 0.5
            
            # 1. Deep DOF (f/16, wide angle)
            metadata_deep = {"aperture": 16.0, "focal_length": 24.0}
            score_deep, expl_deep, _, _ = _calculate_focus_area(self.img, self.gray, base_score, metadata_deep)
            
            # 2. Shallow DOF (f/1.4, portrait)
            metadata_shallow = {"aperture": 1.4, "focal_length": 85.0}
            score_shallow, expl_shallow, _, _ = _calculate_focus_area(self.img, self.gray, base_score, metadata_shallow)
            
            print(f"\nFocus DOF test:")
            print(f"Deep DOF score: {score_deep:.4f}, Explanation: {expl_deep}")
            print(f"Shallow DOF score: {score_shallow:.4f}, Explanation: {expl_shallow}")
            
            # Shallow DOF should be more lenient
            self.assertGreater(score_shallow, score_deep)
            self.assertIn("shallow dof", expl_shallow.lower())


        finally:
            analyzer.g_yolo_model = original_model

if __name__ == '__main__':
    unittest.main()
