"""
Phase 2 helper functions for EXIF-aware metric adjustments.
Provides context-aware scoring based on camera settings and physics.
"""
import numpy as np


def adjust_sharpness_for_aperture(raw_score: float, aperture: float, sensor_size: str, sensor_sizes: dict) -> tuple[float, str]:
    """
    Adjusts sharpness score based on aperture and diffraction physics.
    
    Args:
        raw_score: Raw sharpness score from FFT analysis
        aperture: F-number (e.g., 5.6, 16, 22)
        sensor_size: Sensor size category
        sensor_sizes: Dictionary of sensor size info
    
    Returns:
        (adjusted_score, context_note)
    """
    if aperture is None or sensor_size not in sensor_sizes:
        return raw_score, ""
    
    sensor_info = sensor_sizes[sensor_size]
    diffraction_limit = sensor_info['diffraction_limit']
    
    # Define lens sweet spot and diffraction zones
    if aperture < 2.0:
        # Wide open: expect some edge softness  
        expected_min, expected_max = 0.6, 0.85
        context = f"Wide aperture (f/{aperture:.1f}): edge softness is normal"
    elif aperture <= 8.0:
        # Sweet spot: expect excellent sharpness
        expected_min, expected_max = 0.8, 1.0
        context = f"Optimal aperture (f/{aperture:.1f})"
    elif aperture <= diffraction_limit:
        # Approaching diffraction: slight softness acceptable
        expected_min, expected_max = 0.7, 0.95
        context = f"Good aperture (f/{aperture:.1f})"
    else:
        # Diffraction-limited: softness is physics, not a defect
        beyond_factor = (aperture - diffraction_limit) / 10.0
        softness_penalty = min(beyond_factor * 0.15, 0.3)
        expected_min = max(0.5 - softness_penalty, 0.3)
        expected_max = max(0.8 - softness_penalty, 0.6)
        context = f"Diffraction-limited (f/{aperture:.1f}): softness expected"
    
    # Normalize score to expected range
    if raw_score < expected_min:
        adjusted = raw_score / expected_min * 0.5
    elif raw_score > expected_max:
        adjusted = 1.0
    else:
        range_width = expected_max - expected_min
        if range_width > 0:
            range_position = (raw_score - expected_min) / range_width
            adjusted = 0.5 + (range_position * 0.5)
        else:
            adjusted = 0.75
    
    return float(adjusted), context


def get_camera_dynamic_range_baseline(camera_model: str, iso: int, camera_dr_db: dict) -> float:
    """Returns expected dynamic range in stops for a given camera and ISO."""
    baseline_dr = 12.0
    
    if camera_model:
        for camera_name, dr in camera_dr_db.items():
            if camera_name in camera_model:
                baseline_dr = dr
                break
    
    if iso and iso > 100:
        iso_stops_above_base = np.log2(iso / 100)
        effective_dr = baseline_dr - iso_stops_above_base * 0.5
    else:
        effective_dr = baseline_dr
    
    return max(effective_dr, 8.0)


def get_exposure_tolerance(shutter_speed: float) -> dict:
    """Returns acceptable exposure clipping tolerances based on shutter speed."""
    if shutter_speed is None:
        return {'highlight_clip_tolerance': 0.02, 'shadow_clip_tolerance': 0.02, 'context': 'general'}
    
    if shutter_speed < 0.002:  # < 1/500s (action)
        return {'highlight_clip_tolerance': 0.05, 'shadow_clip_tolerance': 0.10, 'context': 'action'}
    elif shutter_speed > 0.033:  # > 1/30s (long exposure)
        return {'highlight_clip_tolerance': 0.01, 'shadow_clip_tolerance': 0.01, 'context': 'precision'}
    else:
        return {'highlight_clip_tolerance': 0.02, 'shadow_clip_tolerance': 0.02, 'context': 'general'}


def get_expected_focus_area(aperture: float, focal_length: float) -> float:
    """
    Calculates expected in-focus area factor based on aperture and focal length.
    
    Returns a factor (0.0 to 1.0) indicating how much of the scene is expected to be in focus.
    """
    if aperture is None or focal_length is None:
        return 0.5  # Neutral default
    
    # Simplified DOF factor: 
    # DOF is proportional to aperture and inversely proportional to focal_length^2
    # We use a heuristic here to get a sense of "shallow" vs "deep" DOF
    dof_factor = (aperture * 100) / (focal_length**2 + 1)
    
    if dof_factor < 0.1:
        return 0.2  # Extremely shallow (e.g. f/1.4 @ 85mm)
    elif dof_factor < 0.5:
        return 0.4  # Shallow
    elif dof_factor < 1.5:
        return 0.7  # Moderate
    else:
        return 0.9  # Deep (e.g. f/16 @ 24mm)

