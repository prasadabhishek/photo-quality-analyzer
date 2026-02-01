import unittest
import os
import cv2
import numpy as np
from photo_quality_analyzer_core.analyzer import (
    _calculate_sharpness,
    _calculate_exposure,
    _calculate_noise,
    _calculate_dynamic_range,
    evaluate_photo_quality
)

class TestCoreMetrics(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "tests", "data")
        
        self.sharp_path = os.path.join(self.data_dir, "sharp_test.png")
        self.blurry_path = os.path.join(self.data_dir, "blurry_test.png")
        self.noise_path = os.path.join(self.data_dir, "noise_test.png")
        self.white_path = os.path.join(self.data_dir, "white_test.png")

    def test_sharpness_differentiation(self):
        """Verify that sharpness score is significantly higher for sharp vs blurry images."""
        img_sharp = cv2.imread(self.sharp_path, cv2.IMREAD_GRAYSCALE)
        img_blurry = cv2.imread(self.blurry_path, cv2.IMREAD_GRAYSCALE)
        
        score_sharp, _ = _calculate_sharpness(img_sharp)
        score_blurry, _ = _calculate_sharpness(img_blurry)
        
        print(f"Sharpness - Sharp: {score_sharp:.4f}, Blurry: {score_blurry:.4f}")
        self.assertGreater(score_sharp, score_blurry)
        self.assertGreater(score_sharp, 0.001) # Low absolute value for small synthetic assets
        self.assertLess(score_blurry, 0.001)

    def test_exposure_clipping(self):
        """Verify that white/blown-out images get low exposure scores due to clipping."""
        img_white = cv2.imread(self.white_path, cv2.IMREAD_GRAYSCALE)
        score_white, _ = _calculate_exposure(img_white)
        
        print(f"Exposure - Blown White: {score_white:.4f}")
        # Low score expected for multi-zone clipping
        self.assertLess(score_white, 0.5)

    def test_noise_detection(self):
        """Verify that noisy images get lower noise scores (inverse relationship)."""
        img_noise_gray = cv2.imread(self.noise_path, cv2.IMREAD_GRAYSCALE)
        img_noise_color = cv2.imread(self.noise_path, cv2.IMREAD_COLOR)
        
        img_sharp_gray = cv2.imread(self.sharp_path, cv2.IMREAD_GRAYSCALE)
        img_sharp_color = cv2.imread(self.sharp_path, cv2.IMREAD_COLOR)
        
        score_noise_img, _ = _calculate_noise(img_noise_color, img_noise_gray)
        score_sharp_img, _ = _calculate_noise(img_sharp_color, img_sharp_gray)
        
        print(f"Noise Score - Noisy: {score_noise_img:.4f}, Clean: {score_sharp_img:.4f}")
        self.assertLess(score_noise_img, score_sharp_img)

    def test_full_evaluation_flow(self):
        """Verify the main entry point works without AI enabled."""
        result = evaluate_photo_quality(self.sharp_path, enable_subject_detection=False)
        self.assertIn("overallConfidence", result)
        self.assertIn("judgement", result)
        self.assertEqual(result["metadataStatus"], "missing") # Fixed from not_found to missing

if __name__ == "__main__":
    unittest.main()
