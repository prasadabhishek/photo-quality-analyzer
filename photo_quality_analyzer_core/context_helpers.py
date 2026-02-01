"""
Phase 2 helper functions for EXIF-aware metric adjustments.
Provides context-aware scoring based on camera settings and physics.
"""
import numpy as np


def adjust_sharpness_for_aperture(raw_score: float, aperture: float, sensor_size: str, sensor_sizes: dict) -> tuple[float, str]:
    """
    Adjusts the raw sharpness score based on the physics of diffraction.
    
    Science:
    Every lens has a "Diffraction-Limited Aperture" (DLA). Beyond this point,
    the Airy Disk (blur pattern) becomes larger than the sensor's pixels,
    causing natural softness. This function prevents penalizing photographers
    for using narrow apertures (e.g., f/22 for landscapes) where softness
    is a physical inevitability, not a flaw.
    
    Ref: https://www.cambridgeincolour.com/tutorials/diffraction-photography.htm
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


def get_camera_dynamic_range_baseline(base_dr: float, iso: int) -> float:
    """
    Returns the expected effective Dynamic Range (DR) in stops for a given base DR and ISO.
    
    Science:
    Dynamic Range typically peaks at the base ISO (usually ISO 100) and drops
    by approximately 0.5 to 1.0 stops for every 1-stop increase in ISO. 
    This function calculates the expected effective DR for specific shooting conditions.
    
    Ref: https://www.photonstophotos.net/Charts/PDR.htm
    """
    if not base_dr:
        base_dr = 12.0 # Default fallback
    
    if iso and iso > 100:
        # PDR drops approx 0.5 stops per ISO stop increase
        iso_stops_above_base = np.log2(iso / 100)
        effective_dr = base_dr - iso_stops_above_base * 0.5
    else:
        effective_dr = base_dr
    
    return max(effective_dr, 8.0)


def get_exposure_tolerance(shutter_speed: float) -> dict:
    """
    Determines acceptable clipping tolerances based on the "Action Context."
    
    Science:
    - Action shots (fast shutter speed) often involve high-contrast scenes
      where highlight preservation is prioritized over shadows.
    - Long exposures (slow shutter speed) are typically tripod-based 
      landscapes where precise tonal mapping across the whole histogram 
      is required.
    """
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
    Calculates the expected in-focus area factor based on Depth-of-Field (DOF).
    
    Science:
    Depth-of-Field is a function of Aperture, Focal Length, and subject 
    distance. This function uses a heuristic based on the relationship 
    where DOF is proportional to aperture and inversely proportional to 
    the square of the focal length.
    
    Ref: https://en.wikipedia.org/wiki/Depth_of_field
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

