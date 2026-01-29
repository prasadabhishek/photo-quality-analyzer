import unittest
import os
import cv2
from photo_quality_analyzer_core.analyzer import _load_image_with_raw_support

class TestRawSupport(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "tests", "data")
        self.raw_path = os.path.join(self.data_dir, "sample.arw")

    def test_raw_preview_extraction(self):
        """Verify that we can extract a high-resolution preview from a Sony .ARW file."""
        if not os.path.exists(self.raw_path):
            self.skipTest(f"Sample RAW file not found at {self.raw_path}")
            
        img = _load_image_with_raw_support(self.raw_path)
        
        self.assertIsNotNone(img, "Failed to load RAW image preview")
        h, w, c = img.shape
        print(f"RAW Preview Dimensions: {w}x{h}")
        
        # We upgraded the logic to pick the largest preview. 
        # For Sony, this should be at least 1080p width or equivalent.
        self.assertGreaterEqual(w, 1024, "Extracted preview resolution is too low (expected > 1024px width)")
        self.assertEqual(c, 3, "Image should have 3 color channels")

if __name__ == "__main__":
    unittest.main()
