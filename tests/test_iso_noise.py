"""
Unit tests for ISO-adaptive noise normalization.
Tests the fix for the bug where clean ISO 400 images scored 0.0 on noise.
"""
import unittest
import os
import sys
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../photo-quality-analyzer')))

from photo_quality_analyzer_core.analyzer import _calculate_noise


class TestISOAdaptiveNoise(unittest.TestCase):
    """Test ISO-adaptive noise normalization."""
    
    def setUp(self):
        """Create test images with known noise floor."""
        # Simulate a clean image with natural texture variance
        np.random.seed(42)
        self.clean_with_texture = np.random.normal(128, 15, (1000, 1000)).astype(np.uint8)
    
    def test_iso_100_clean_image(self):
        """Clean image at ISO 100 should score high."""
        metadata = {'iso': 100}
        score, explanation = _calculate_noise(self.clean_with_texture, metadata)
        
        # At ISO 100, this should be considered clean
        self.assertGreater(score, 0.2, f"ISO 100 clean image should score >0.2, got {score}")
        self.assertIn("noise", explanation.lower())
    
    def test_iso_400_clean_image(self):
        """Clean image at ISO 400 should score reasonably (NOT 0.0)."""
        metadata = {'iso': 400}
        score, explanation = _calculate_noise(self.clean_with_texture, metadata)
        
        # This is the critical fix: should NOT be 0.0
        self.assertGreater(score, 0.4, f"ISO 400 clean image should score >0.4, got {score}")
        self.assertNotEqual(score, 0.0, "ISO 400 should not score 0.0 with new normalization")
    
    def test_iso_1600_moderate_noise(self):
        """Moderate noise at ISO 1600 should be more lenient."""
        metadata = {'iso': 1600}
        score, explanation = _calculate_noise(self.clean_with_texture, metadata)
        
        # Higher ISO gets more lenient normalization
        self.assertGreater(score, 0.6, f"ISO 1600 with moderate noise should score >0.6, got {score}")
    
    def test_iso_6400_high_noise(self):
        """High noise at ISO 6400 should be very lenient."""
        # Simulate noisier image
        noisy_image = np.random.normal(128, 30, (1000, 1000)).astype(np.uint8)
        metadata = {'iso': 6400}
        score, explanation = _calculate_noise(noisy_image, metadata)
        
        # Even noisy images at high ISO should get reasonable scores
        self.assertGreater(score, 0.3, f"ISO 6400 noisy image should score >0.3, got {score}")
        self.assertIn("ISO 6400", explanation, "Explanation should mention the high ISO")
    
    def test_no_metadata_fallback(self):
        """Images without ISO metadata should use fallback normalization."""
        metadata = None
        score, explanation = _calculate_noise(self.clean_with_texture, metadata)
        
        # Fallback to 400.0 normalization factor
        self.assertGreater(score, 0.0, "Should not score 0.0 even without metadata")
    
    def test_empty_metadata_fallback(self):
        """Empty metadata dict should use fallback."""
        metadata = {}
        score, explanation = _calculate_noise(self.clean_with_texture, metadata)
        
        self.assertGreater(score, 0.0, "Should not score 0.0 with empty metadata")
    
    def test_explanation_includes_iso_context(self):
        """Explanations should be context-aware for high ISO."""
        metadata = {'iso': 3200}
        score, explanation = _calculate_noise(self.clean_with_texture, metadata)
        
        # Should mention ISO in explanation if noise is present
        if score < 0.8:
            self.assertIn("ISO", explanation, "Explanation should provide ISO context")


if __name__ == '__main__':
    unittest.main()
