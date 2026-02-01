import unittest
import os
import numpy as np
import cv2
import json
import shutil
import tempfile
from photo_quality_analyzer_core.analyzer import (
    get_camera_data,
    detect_sensor_size,
    generate_color_palette,
    _calculate_color_balance,
    _calculate_composition,
    create_xmp_sidecar,
    load_camera_database,
    evaluate_photo_quality
)

class TestLibraryCompleteness(unittest.TestCase):
    """
    Comprehensive test suite to ensure open-source readiness and 
    independent library functionality.
    """

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for file-based tests
        cls.test_dir = tempfile.mkdtemp()
        cls.dummy_image_path = os.path.join(cls.test_dir, "test_image.jpg")
        
        # Create a dummy image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add some color and "subjects"
        cv2.rectangle(img, (90, 90), (110, 110), (0, 255, 0), -1) # Green subject at Rule of Thirds power point
        cv2.imwrite(cls.dummy_image_path, img)

    @classmethod
    def tearDownClass(cls):
        # Cleanup temp directory
        shutil.rmtree(cls.test_dir)

    def test_camera_database_lookup(self):
        """Verify that known camera models and aliases resolve correctly."""
        # 1. Direct match
        data_a1 = get_camera_data("Sony A1")
        self.assertIsNotNone(data_a1)
        self.assertEqual(data_a1["sensor_size"], "full_frame")
        self.assertEqual(data_a1["dr"], 14.5)

        # 2. Alias match
        data_a7rv = get_camera_data("ILCE-7RM5")
        self.assertIsNotNone(data_a7rv)
        self.assertEqual(data_a7rv["sensor_size"], "full_frame")

        # 3. Best (longest) match test - D7000 vs D700
        # In the DB, D7000 is APS-C, D700 is Full Frame (D700 is not in DB yet? let's check)
        # Actually I added D7000 recently.
        data_d7000 = get_camera_data("NIKON D7000")
        self.assertIsNotNone(data_d7000)
        self.assertEqual(data_d7000["sensor_size"], "aps_c")

    def test_hardware_heuristics(self):
        """Verify that unknown models fallback to correct heuristics."""
        # 1. Unknown Sony A-series (should be Full Frame)
        size_a7x = detect_sensor_size("Sony A7Z MK IV")
        self.assertEqual(size_a7x, "full_frame")

        # 2. Unknown Fuji X-series (should be APS-C)
        size_xtx = detect_sensor_size("Fujifilm X-T99")
        self.assertEqual(size_xtx, "aps_c")

        # 3. Unknown Olympus (should be Micro Four Thirds)
        size_om = detect_sensor_size("Olympus E-M99")
        self.assertEqual(size_om, "micro_four_thirds")

    def test_color_palette_and_balance(self):
        """Verify palette extraction and color balance reporting."""
        # 1. Palette extraction
        palette = generate_color_palette(self.dummy_image_path, num_colors=3)
        self.assertEqual(len(palette), 3)
        self.assertTrue(all(isinstance(c, str) and c.startswith('#') for c in palette))

        # 2. Color balance - Neutral image
        img_neutral = np.full((100, 100, 3), 128, dtype=np.uint8)
        score, expl = _calculate_color_balance(img_neutral)
        self.assertGreater(score, 0.9)
        self.assertIn("neutral", expl.lower())

    def test_composition_rule_of_thirds(self):
        """Verify composition scoring for subject placement."""
        # Mock detection at a power point (1/3, 1/3)
        # Image is 300x300, power point is (100, 100)
        detections = [[95, 95, 105, 105]] 
        score, expl = _calculate_composition((300, 300, 3), detections)
        self.assertGreater(score, 0.7)
        self.assertIn("Rule of Thirds", expl)

        # Centered detection
        detections_center = [[145, 145, 155, 155]]
        score_c, expl_c = _calculate_composition((300, 300, 3), detections_center)
        self.assertLess(score_c, score)

    def test_xmp_sidecar_generation(self):
        """Verify that XMP files are created with valid content."""
        xmp_path = os.path.splitext(self.dummy_image_path)[0] + ".xmp"
        if os.path.exists(xmp_path):
            os.remove(xmp_path)
            
        create_xmp_sidecar(self.dummy_image_path, "Rejected", 0.3)
        self.assertTrue(os.path.exists(xmp_path))
        
        with open(xmp_path, 'r') as f:
            content = f.read()
            self.assertIn('xmp:Rating="1"', content) # 0.3 * 5 = 1.5 -> 1
            self.assertIn('xmp:Label="Rejected"', content)

    def test_independent_library_entrypoint(self):
        """Confirm evaluate_photo_quality works in standalone mode."""
        results = evaluate_photo_quality(self.dummy_image_path, enable_subject_detection=False)
        self.assertIn("overallConfidence", results)
        self.assertIn("technicalScore", results)
        self.assertIn("judgement", results)
        self.assertIsInstance(results["metrics"], dict)

if __name__ == '__main__':
    unittest.main()
