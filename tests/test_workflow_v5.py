
import unittest
import os
import shutil
import tempfile
import cv2
import numpy as np
from photo_quality_analyzer_core.analyzer import process_folder, evaluate_photo_quality

class TestWorkflowEnhancements(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.assets_dir = os.path.join(self.test_dir, "assets")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.assets_dir)
        os.makedirs(self.output_dir)
        
        # Create Dummy Images
        # 1. High Quality Image (Green Square)
        self.good_img_path = os.path.join(self.assets_dir, "good.jpg")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), -1) 
        cv2.imwrite(self.good_img_path, img)

        # 2. Low Quality Image (Black Noise)
        self.bad_img_path = os.path.join(self.assets_dir, "bad.jpg")
        noise = np.random.randint(0, 50, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(self.bad_img_path, noise)
        
        # 3. TIFF Support Test
        self.tiff_path = os.path.join(self.assets_dir, "test.tiff")
        cv2.imwrite(self.tiff_path, img) # Re-save good image as TIFF

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tiff_support(self):
        """Verify that TIFF files are processed."""
        # We manually process just the TIFF
        result = evaluate_photo_quality(self.tiff_path, enable_subject_detection=False)
        self.assertIsNotNone(result)
        self.assertIn("judgement", result)

    def test_threshold_move_selects(self):
        """Verify --min_conf moves good images to 'selects'."""
        # Use a low threshold so the good image passes
        process_folder(self.assets_dir, verbose=False, move_files=True, min_conf_threshold=0.1)
        
        # Debug: List what's in the assets directory
        selects_dir = os.path.join(self.assets_dir, "selects")
        if not os.path.exists(selects_dir):
            # Print directory contents for debugging
            print(f"Contents of {self.assets_dir}: {os.listdir(self.assets_dir)}")
            self.fail(f"selects/ directory was not created in {self.assets_dir}")
        
        selects_path = os.path.join(selects_dir, "good.jpg")
        if not os.path.exists(selects_path):
            # Print selects directory contents for debugging  
            print(f"Contents of selects/: {os.listdir(selects_dir)}")
            self.fail(f"good.jpg was not moved to selects/")
        
        self.assertTrue(os.path.exists(selects_path))
        
    def test_threshold_move_rejects(self):
        """Verify --min_conf moves bad images to 'rejects'."""
        # Use a high threshold so everything fails (even good dummy might fail strict criteria, but bad surely will)
        # Note: We need to reset the good image location as it might have moved in previous test if we don't mock/isolate
        # But here setUp creates fresh dir every time
        
        process_folder(self.assets_dir, verbose=False, move_files=True, min_conf_threshold=0.99)
        
        # Debug: List what's in the assets directory
        rejects_dir = os.path.join(self.assets_dir, "rejects")
        if not os.path.exists(rejects_dir):
            print(f"Contents of {self.assets_dir}: {os.listdir(self.assets_dir)}")
            self.fail(f"rejects/ directory was not created in {self.assets_dir}")
        
        rejects_path = os.path.join(rejects_dir, "bad.jpg")
        if not os.path.exists(rejects_path):
            print(f"Contents of rejects/: {os.listdir(rejects_dir)}")
            self.fail(f"bad.jpg was not moved to rejects/")
        
        self.assertTrue(os.path.exists(rejects_path))

if __name__ == '__main__':
    unittest.main()
