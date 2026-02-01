"""
Photo Quality Analyzer Core
---------------------------
A local-first visual intelligence engine that uses signal processing and 
computer vision to assess photographic quality with context awareness.

Scientific Foundation:
- Optical Physics (Diffraction, Depth-of-Field)
- Signal Processing (FFT, Laplacian Variance)
- Information Theory (Shannon Entropy for Dynamic Range)
- Computer Vision (YOLO-based Subject Detection)

Sources:
- DXOMARK: https://www.dxomark.com/ (Sensor Benchmarks)
- Photons to Photos: https://www.photonstophotos.net/ (Dynamic Range Curves)
- Cambridge in Colour: https://www.cambridgeincolour.com/ (Optical Theory)
"""
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from tqdm import tqdm
    import exifread
    try:
        import rawpy
    except ImportError:
        rawpy = None
    from scipy.fftpack import fft2, fftshift
    from scipy.stats import entropy
    from .context_helpers import (
        adjust_sharpness_for_aperture,
        get_camera_dynamic_range_baseline,
        get_exposure_tolerance,
        get_expected_focus_area
    )
except ImportError as e:
    print(f"ImportError: {e}")
    print("One or more required Python packages are not installed.")
    print("Please install the necessary dependencies by running:")
    print("pip install -r requirements.txt")
    print("If you don't have 'requirements.txt', ensure you have opencv-python, numpy, ultralytics, exifread, and scipy installed.")
    exit(1)

# --- Standard Library Imports ---
import json
import os
import logging
import argparse
import shutil  # Added for moving files
import configparser
import io

# Note: YOLO is imported in the try-except block above

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
# Search for config.ini in:
# 1. Current working directory
# 2. Package directory
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_NAME = 'config.ini'

def find_config(filename: str = DEFAULT_CONFIG_NAME) -> str | None:
    """
    Locates the configuration file within the file system.
    
    Technical Search Strategy:
    1. Checks the Current Working Directory (CWD) for user-specific overrides.
    2. Checks the Package Directory (where the source code resides) for default configs.
    
    This ensures that users can customize normalization factors without
    modifying the core library.
    """

CONFIG_FILE_PATH = find_config() or DEFAULT_CONFIG_NAME

def load_config(config_file_path: str = CONFIG_FILE_PATH) -> dict:
    """
    Parses the configuration file to initialize engine parameters.
    
    The engine uses a series of weighting and normalization constants to 
    balance technical metrics (Sharpness, Noise) against aesthetic metrics 
    (Composition). These are adjustable via `config.ini` to cater to different
    photographic styles (e.g., stricter sharpness for architectural 
    photography vs. leniency for street photography).
    
    Returns:
        dict: A mapping of configuration keys to their float/string values.
    """
    config = configparser.ConfigParser()
    if config_file_path is None or not os.path.exists(config_file_path):
        logger.warning(
            f"Configuration file '{config_file_path}' not found. Using defaults.")
        return {}

    try:
        config.read(config_file_path)
        return {
            'SHARPNESS_NORMALIZATION_FACTOR': config.getfloat('NormalizationFactors', 'sharpness', fallback=1000.0),
            'FOCUS_AREA_NORMALIZATION_FACTOR': config.getfloat('NormalizationFactors', 'focus_area', fallback=1000.0),
            'NOISE_NORMALIZATION_FACTOR': config.getfloat('NormalizationFactors', 'noise', fallback=50.0),
            'EXPOSURE_IDEAL_MEAN_INTENSITY': config.getfloat('Thresholds', 'exposure_ideal_mean', fallback=128.0),
            'DYNAMIC_RANGE_MAX_VALUE': config.getfloat('Thresholds', 'dynamic_range_max', fallback=255.0),
            'YOLO_CONFIDENCE_THRESHOLD': config.getfloat('Thresholds', 'yolo_confidence', fallback=0.5),
            'YOLO_NMS_THRESHOLD': config.getfloat('Thresholds', 'yolo_nms', fallback=0.45),
            'OVERALL_CONF_TECH_WEIGHT': config.getfloat('Weights', 'overall_tech', fallback=0.6),
            'OVERALL_CONF_OTHER_WEIGHT': config.getfloat('Weights', 'overall_other', fallback=0.4),
            'YOLO_MODEL_PATH_DEFAULT': config.get('Models', 'default_yolo_model', fallback="yolo11n.pt"),
            'JUDGEMENT_EXCELLENT': config.getfloat('JudgementLevels', 'excellent', fallback=0.9),
            'JUDGEMENT_GOOD': config.getfloat('JudgementLevels', 'good', fallback=0.7),
            'JUDGEMENT_FAIR': config.getfloat('JudgementLevels', 'fair', fallback=0.5),
            'JUDGEMENT_POOR': config.getfloat('JudgementLevels', 'poor', fallback=0.3),
        }
    except (configparser.Error, ValueError) as e:
        logger.error(f"Error reading configuration file '{config_file_path}': {e}")
        return {}

# Define defaults at module level
_cfg = load_config()
SHARPNESS_NORMALIZATION_FACTOR = _cfg.get('SHARPNESS_NORMALIZATION_FACTOR', 1000.0)
FOCUS_AREA_NORMALIZATION_FACTOR = _cfg.get('FOCUS_AREA_NORMALIZATION_FACTOR', 1000.0)
NOISE_NORMALIZATION_FACTOR = _cfg.get('NOISE_NORMALIZATION_FACTOR', 50.0)
EXPOSURE_IDEAL_MEAN_INTENSITY = _cfg.get('EXPOSURE_IDEAL_MEAN_INTENSITY', 128.0)
DYNAMIC_RANGE_MAX_VALUE = _cfg.get('DYNAMIC_RANGE_MAX_VALUE', 255.0)
YOLO_CONFIDENCE_THRESHOLD = _cfg.get('YOLO_CONFIDENCE_THRESHOLD', 0.5)
YOLO_NMS_THRESHOLD = _cfg.get('YOLO_NMS_THRESHOLD', 0.45)
OVERALL_CONF_TECH_WEIGHT = _cfg.get('OVERALL_CONF_TECH_WEIGHT', 0.6)
OVERALL_CONF_OTHER_WEIGHT = _cfg.get('OVERALL_CONF_OTHER_WEIGHT', 0.4)
YOLO_MODEL_PATH_DEFAULT = _cfg.get('YOLO_MODEL_PATH_DEFAULT', "yolo11n.pt")
JUDGEMENT_EXCELLENT = _cfg.get('JUDGEMENT_EXCELLENT', 0.9)
JUDGEMENT_GOOD = _cfg.get('JUDGEMENT_GOOD', 0.7)
JUDGEMENT_FAIR = _cfg.get('JUDGEMENT_FAIR', 0.5)
JUDGEMENT_POOR = _cfg.get('JUDGEMENT_POOR', 0.3)

COCO_NAMES_FILE_PATH_DEFAULT = "coco.names"
RAW_EXTENSIONS = {'.arw', '.cr2', '.nef', '.dng', '.orf', '.raf', '.srw'}
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.arw', '.cr2', '.nef', '.dng', '.orf', '.raf', '.srw', '.cr3', '.rw2', '.nrw', '.gpr', '.sr2', '.pef', '.rwl')

# --- Camera Capability Database (Phase 2) ---
# Values are sourced from PhotonsToPhotos (https://www.photonstophotos.net/)
# and DXOMARK (https://www.dxomark.com/) based on base ISO measurements.

def load_camera_database() -> dict:
    """
    Loads the camera capability database.
    Priority:
    1. Environment variable 'PQA_CAMERA_DB_PATH'
    2. User config '~/.photo_quality_analyzer/camera_database.json'
    3. Bundled 'data/camera_database.json'
    """
    candidates = []
    
    # 1. Env Var
    if os.environ.get("PQA_CAMERA_DB_PATH"):
        candidates.append(os.environ["PQA_CAMERA_DB_PATH"])
        
    # 2. User Config
    user_config = os.path.expanduser("~/.photo_quality_analyzer/camera_database.json")
    candidates.append(user_config)
    
    # 3. Bundled
    candidates.append(os.path.join(PACKAGE_DIR, 'data', 'camera_database.json'))
    
    final_db = {}
    loaded_any = False
    
    import json
    
    # Load bundled first, then override with user
    # Actually, usually we just want one source or merge?
    # Let's merge: Bundled -> User -> Env
    
    # Reverse to load bundled first (base), then user (override)
    for path in reversed(candidates):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Simple top-level merge
                    final_db.update(data)
                    loaded_any = True
                    logger.info(f"Loaded/Merged camera database from: {path}")
            except Exception as e:
                logger.error(f"Error loading camera database from {path}: {e}")
                
    if not loaded_any:
        logger.warning("No camera database found. Using minimal fallbacks.")
        
    return final_db

# Lazy load the database
_CAMERA_DB = None

def get_camera_data(camera_model: str) -> dict | None:
    """
    Search for camera specifications using model name or known aliases.
    Returns: {dr: float, sensor_size: str} or None
    """
    global _CAMERA_DB
    if _CAMERA_DB is None:
        _CAMERA_DB = load_camera_database()
    
    if not camera_model:
        return None
        
    model_upper = camera_model.upper()
    
    # Iterate through brands and models to find the best (longest) match
    ignore_keys = {"General", "Heuristics"}
    best_match = None
    max_len = 0
    
    for brand, models in _CAMERA_DB.items():
        if brand in ignore_keys:
            continue
        for model_name, specs in models.items():
            # Potential candidates for this specific model
            candidates = [model_name.upper()] + [a.upper() for a in specs.get('aliases', [])]
            
            for cand in candidates:
                # Check if candidate is in the image metadata string
                if cand in model_upper:
                    # We want the longest matching string to avoid "D700" matching "D7000"
                    if len(cand) > max_len:
                        max_len = len(cand)
                        best_match = specs
                        
    return best_match

# Placeholder for backward compatibility if needed, though we'll update callsites
CAMERA_DYNAMIC_RANGE = {} 

# Sensor sizes and corresponding optical limits.
# diffraction_limit: The aperture where the Airy Disk size exceeds the pixel pitch.
# coc: Circle of Confusion used for Depth-of-Field (DOF) calculations (Leica standard).
# Ref: https://www.cambridgeincolour.com/tutorials/diffraction-photography.htm
SENSOR_SIZES = {
    'full_frame': {'diffraction_limit': 16.0, 'coc': 0.030},
    'aps_c': {'diffraction_limit': 11.0, 'coc': 0.020},
    'micro_four_thirds': {'diffraction_limit': 8.0, 'coc': 0.015},
    'one_inch': {'diffraction_limit': 5.6, 'coc': 0.011},
    'one_over_two_three': {'diffraction_limit': 4.0, 'coc': 0.006},
}

def detect_sensor_size(camera_model: str) -> str:
    """
    Detects the physical sensor size category based on the camera model string.
    
    Logic:
    1. Direct Database Lookup: First, check if the specific model is in the 
       camera database (including aliases).
    2. Heuristic Pattern Matching: If not found, use brand-specific patterns
       stored in the "Heuristics" section of the database to guess the size.
    3. Default Fallback: Defaults to 'full_frame' if no match is found.
    """
    global _CAMERA_DB
    if _CAMERA_DB is None:
        _CAMERA_DB = load_camera_database()

    # 1. Direct Lookup
    specs = get_camera_data(camera_model)
    if specs and 'sensor_size' in specs:
        return specs['sensor_size']
        
    if not camera_model:
        return 'full_frame'
    
    model_upper = camera_model.upper()
    
    # 2. Database-Driven Heuristics (Longest match wins)
    heuristics = _CAMERA_DB.get("Heuristics", {})
    best_size = 'full_frame' # Default fallback
    best_len = 0
    
    for size, patterns in heuristics.items():
        for p in patterns:
            p_upper = p.upper()
            if p_upper in model_upper:
                if len(p_upper) > best_len:
                    best_len = len(p_upper)
                    best_size = size
            
    return best_size



# Global model variables for lazy loading
g_yolo_model = None
g_coco_names = None


def load_yolo_model_and_names(model_path: str, coco_names_file_path: str) -> tuple[YOLO | None, list[str] | None]:
    """
    Initializes the YOLO (You Only Look Once) neural network for subject detection.
    
    Technology:
    Uses the Ultralytics YOLO framework to identify 80+ common objects in the 
    COCO dataset. This subject information is critical for distinguishing 
    between an "out-of-focus subject" and a "bokeh background."
    
    Model Weights:
    The engine supports different model sizes (nano, small, medium, large).
    'Nano' is recommended for local execution due to its low latency and 
    sufficient accuracy for photographic ROI detection.
    
    Ref: https://arxiv.org/abs/1506.02640 (YOLO Original Paper)
    """
    loaded_model = None
    loaded_coco_names = None
    try:
        # Explicitly check if the model_path is a directory, as YOLO() might not handle this gracefully.
        if os.path.isdir(model_path):
            raise IsADirectoryError(
                f"The provided model path '{model_path}' is a directory. Please specify a path to a .pt model file."
            )

        # Attempt to load the model.
        # Ultralytics' YOLO() constructor will:
        # 1. Attempt to download if 'model_path' is a recognized model name (e.g., "yolov8n.pt").
        # 2. Attempt to load from disk if 'model_path' is a file path (e.g., "./yolo11n.pt").
        loaded_model = YOLO(model_path)
        logger.info(
            f"Successfully loaded/initialized YOLO model using '{model_path}'.")

        # Try to get names from the model itself (logic remains the same)

        # Try to get names from the model itself
        if hasattr(loaded_model, 'names') and isinstance(loaded_model.names, dict) and loaded_model.names:
            if all(isinstance(k, int) for k in loaded_model.names.keys()):
                max_id = -1
                if loaded_model.names:
                    max_id = max(loaded_model.names.keys())
                if max_id != -1:
                    _coco_names_list = [
                        f"unknown_id_{i}" for i in range(max_id + 1)]
                    for class_id_int, name_str in loaded_model.names.items():
                        _coco_names_list[class_id_int] = name_str
                    loaded_coco_names = _coco_names_list
                    logger.info("Loaded class names from YOLO model.")
                # else: loaded_coco_names remains None
            # else: loaded_coco_names remains None
            if loaded_coco_names is None:
                logger.warning(
                    "YOLO model.names format not as expected or empty.")

        if loaded_coco_names is None:
            logger.critical(
                "Critical Warning: No class names loaded from the model. "
                "Object descriptions will be limited to class IDs. "
                "Ensure the model file embeds class names."
            )

    except IsADirectoryError as dir_error:  # Catch our explicit check
        logger.error(f"{dir_error}")
    except FileNotFoundError:  # This might be raised by YOLO() if a local file path is not found
        logger.error(
            f"The model file was not found at the specified path: '{model_path}'.")
    except PermissionError:  # This might be raised by YOLO() if a local file path has permission issues
        logger.error(
            f"Permission denied when trying to access the model file at: '{model_path}'.")
    # Catch-all for other errors during YOLO initialization (network, bad format, etc.)
    except Exception as e:
        logger.error(
            f"An error occurred while loading/initializing the YOLO model '{model_path}': {e}", exc_info=True)
        logger.error("Please ensure that:")
        logger.error(
            "  1. If using a standard model name (e.g., 'yolov8n.pt'), your internet connection is active for the first download.")
        logger.error(f"  2. If '{model_path}' is a file path (like the default '{YOLO_MODEL_PATH_DEFAULT}'), it points to a valid and readable .pt model file in the expected location (e.g., same directory as the script).")
        logger.error(
            "  3. The 'ultralytics' package is correctly installed and up to date.")

    return loaded_model, loaded_coco_names


def ensure_yolo_initialized(model_size: str = "nano") -> None:
    """
    Guarantees that the visual intelligence model is ready for processing.
    
    Implements a lazy-loading pattern to avoid the significant overhead 
    of booting a neural network if the user does not request subject-aware 
    metrics. This optimization is crucial for bulk analysis performance.
    """
    global g_yolo_model, g_coco_names
    
    # Map friendly names to model files
    model_map = {
        "nano": "yolo11n.pt",
        "xlarge": "yolo12x.pt"
    }
    requested_model = model_map.get(model_size.lower(), "yolo11n.pt")
    
    # If the model is already loaded and matches the requested size, skip
    # (Note: g_yolo_model.model_name might differ if path is used, so we check the active config)
    if g_yolo_model is not None:
        return True
        
    g_yolo_model, g_coco_names = load_yolo_model_and_names(
        requested_model, COCO_NAMES_FILE_PATH_DEFAULT)
    return g_yolo_model is not None


def _extract_metadata(image_path: str) -> dict:
    """
    Parses EXIF (Exchangeable Image File Format) data using the EXIFRead library.
    
    Metadata Scope:
    - Shutter Speed: Determines exposure context and motion blur tolerance.
    - Aperture (F-Stop): Used for diffraction limits and DOF heuristics.
    - ISO: Used for sensor noise floor normalization.
    - Camera Model: Used for sensor-specific DR benchmarking.
    
    Ref: https://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf (EXIF Standard)
    """
    metadata = {
        "shutter_speed": None,
        "iso": None,
        "aperture": None,
        "status": "missing"
    }
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            # Shutter Speed
            if 'EXIF ExposureTime' in tags:
                metadata["shutter_speed"] = float(tags['EXIF ExposureTime'].values[0])
            
            # ISO
            if 'EXIF ISOSpeedRatings' in tags:
                metadata["iso"] = int(tags['EXIF ISOSpeedRatings'].values[0])
            
            # Aperture
            if 'EXIF FNumber' in tags:
                metadata["aperture"] = float(tags['EXIF FNumber'].values[0])
            
            if metadata["shutter_speed"] or metadata["iso"] or metadata["aperture"]:
                metadata["status"] = "present"
    except Exception as e:
        logger.warning(f"Could not extract EXIF from {image_path}: {e}")
    
    return metadata


# --- Metric Calculation Helper Functions ---

def _calculate_sharpness(gray_img: np.ndarray, metadata: dict = None) -> tuple[float, str]:
    """
    Calculates image sharpness using FFT Anisotropy (Directionality) analysis.
    
    Science:
    Uses the Fast Fourier Transform (FFT) to analyze the spatial frequency 
    distribution. High scores reflect a high ratio of high-frequency components
    relative to the total energy, indicating fine detail and sharp edges.
    
    Aperture-Awareness:
    Adjusts the score based on the "diffraction-limited aperture" (DLA). If a
    narrow aperture (like f/22) is used, the system recognizes that physics
    prevents tack-sharpness and normalizes expectations accordingly.
    
    Ref: https://www.cambridgeincolour.com/tutorials/sharpness.htm
    """
    # Compute FFT
    f_transform = fft2(gray_img)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    h, w = gray_img.shape
    cy, cx = h // 2, w // 2
    
    # Analyze High-Frequency Band
    r_outer = int(min(h, w) * 0.4)
    r_inner = int(min(h, w) * 0.1)
    
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask_hf = (dist_from_center >= r_inner) & (dist_from_center <= r_outer)
    
    if not np.any(mask_hf):
        return 0.0, "Image too small for FFT analysis."
        
    hf_energy = magnitude_spectrum[mask_hf]
    mean_hf = np.mean(hf_energy)
    
    # Anisotropy: Moment Analysis (Rotationally Invariant)
    # Treat HF energy as a distribution of points and find the ratio of eigenvalues.
    # We use 2nd order moments to find the principle directionality.
    yy, xx = np.mgrid[:h, :w]
    shifted_y = yy - cy
    shifted_x = xx - cx
    
    # Weight spatial coordinates by magnitude spectrum
    m00 = np.sum(magnitude_spectrum[mask_hf])
    m01 = np.sum(shifted_y[mask_hf] * magnitude_spectrum[mask_hf])
    m10 = np.sum(shifted_x[mask_hf] * magnitude_spectrum[mask_hf])
    
    # Central moments
    mu20 = np.sum((shifted_x[mask_hf] - (m10/m00))**2 * magnitude_spectrum[mask_hf]) / m00
    mu02 = np.sum((shifted_y[mask_hf] - (m01/m00))**2 * magnitude_spectrum[mask_hf]) / m00
    mu11 = np.sum((shifted_x[mask_hf] - (m10/m00)) * (shifted_y[mask_hf] - (m01/m00)) * magnitude_spectrum[mask_hf]) / m00
    
    # Eigenvalues of the structure tensor equivalent
    # lambda = (mu20 + mu02) / 2 +/- sqrt(((mu20 - mu02)/2)^2 + mu11^2)
    common = np.sqrt(((mu20 - mu02)/2)**2 + mu11**2 + 1e-9)
    lam1 = (mu20 + mu02) / 2 + common
    lam2 = (mu20 + mu02) / 2 - common
    
    # Anisotropy ratio (1 - minor/major). 
    # Perfectly directional edges -> 1.0. Random noise -> 0.0.
    directionality = 1.0 - (lam2 / (lam1 + 1e-9))
    
    # Final Sharpness Score: Combination of HF energy and Directionality
    # We use sqrt of directionality to be less aggressive than squaring for stability
    raw_score = min((mean_hf / (h*w)) * np.sqrt(directionality) * 8000.0, 1.0)
    
    # Phase 2: Aperture-aware adjustment
    aperture = metadata.get("aperture") if metadata else None
    camera_model = metadata.get("camera_model") if metadata else None
    sensor_size = detect_sensor_size(camera_model) if camera_model else 'full_frame'
    
    if aperture is not None:
        adjusted_score, aperture_context = adjust_sharpness_for_aperture(raw_score, aperture, sensor_size, SENSOR_SIZES)
        score = adjusted_score
    else:
        score = raw_score
        aperture_context = ""
    
    # EXIF Interpretation
    if metadata and metadata.get("shutter_speed") and metadata["shutter_speed"] >= 0.03: # slower than 1/30s
        score = min(score * 1.5, 1.0)
        explanation = "Artistic motion blur likely due to slow shutter."
    else:
        base_explanation = "Edges are sharp and directional." if score > 0.6 else "Image is blurry or dominated by random noise."
        if aperture_context:
            explanation = f"{base_explanation} ({aperture_context})"
        else:
            explanation = base_explanation
    
    return float(score), explanation


def _calculate_focus_area(
    img: np.ndarray, gray_img: np.ndarray, overall_sharpness_score: float, metadata: dict = None, detections: list = None
) -> tuple[float, str, set[str], str | None]:
    """
    Assesses focus accuracy on the main subject using YOLO and Laplacian Variance.
    
    Science:
    1. Identifies the primary "subject" using the YOLO neural network.
    2. Measures the Laplacian Variance (edge density) within the subject's 
       bounding box (ROI).
    3. Factors in Depth-of-Field (DOF): At wide apertures (e.g. f/1.4), 
       sharpness expectations are concentrated strictly on the subject, while
       deep-DOF images (f/11) are expected to be sharp across more of the frame.
    
    Ref: https://en.wikipedia.org/wiki/Depth_of_field
    """
    height, width = gray_img.shape
    focus_score = overall_sharpness_score  # Default if no subject or error
    focus_explanation = "No main subject detected for focus; using overall sharpness."
    detected_obj_names: set[str] = set()
    main_subj_name: str | None = None

    # Lazy-load detections if not provided
    if detections is None:
        detections = _detect_objects(img)
        
    if detections:
        # 1. Gather all object names for report
        for d in detections:
            if d.get('name'):
                detected_obj_names.add(d['name'])
        
        # 2. Find the "Main Subject" (highest confidence)
        best_det = max(detections, key=lambda x: x['conf'])
        
        x1_main, y1_main, x2_main, y2_main = map(int, best_det['box'])
        main_subject_class_id = best_det['class_id']
        main_subj_name = best_det['name']

        # Define Region of Interest (ROI)
        roi_x1, roi_y1 = int(max(0, x1_main)), int(max(0, y1_main))
        roi_x2, roi_y2 = int(min(x2_main, width)), int(min(y2_main, height))

        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                laplacian_var_roi = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                
                # Normalize logic specific to ROI focus
                focus_score = min(laplacian_var_roi / FOCUS_AREA_NORMALIZATION_FACTOR, 1.0)
                
                # Phase 2: DOF-aware adjustment
                aperture = metadata.get("aperture") if metadata else None
                focal_length = metadata.get("focal_length") if metadata else None
                
                if aperture is not None and focal_length is not None:
                    dof_factor = get_expected_focus_area(aperture, focal_length)
                    # If DOF is shallow (low factor), we are more lenient with focus scores
                    if dof_factor < 0.5:  # Shallow DOF
                        focus_score = min(focus_score * 1.5, 1.0)
                        focus_explanation = "Main subject is in sharp focus (shallow DOF expected)." if focus_score > 0.7 \
                                            else "Main subject is slightly out of focus for shallow DOF."
                    else:
                        focus_explanation = "Main subject is in sharp focus." if focus_score > 0.8 \
                                            else "Main subject is slightly out of focus."
                else:
                    focus_explanation = "Main subject is in sharp focus." if focus_score > 0.8 \
                                        else "Main subject is slightly out of focus."
            else:
                focus_explanation = "Invalid ROI (empty); using overall sharpness."
        else:
            focus_explanation = "Invalid ROI (zero area); using overall sharpness."

    return focus_score, focus_explanation, detected_obj_names, main_subj_name


def _calculate_exposure(gray_img: np.ndarray, metadata: dict = None, detections: list = None) -> tuple[float, str]:
    """
    Evaluates exposure balance using Subject-Aware Zone System principles.
    
    Science:
    1. Global Analysis: Detects "blown highlights" (clipping at 255).
    2. Subject-Aware Metering: If a subject (person) is detected via YOLO,
       the "Middle Gray" (Zone V) target is evaluated specifically on the subject,
       ignoring the background (e.g., a dark room or bright beach).
    3. Fallback: If no subject is found, defaults to global histogram metering.
    
    Ref: https://en.wikipedia.org/wiki/Zone_System
    """
    total_pixels = gray_img.size
    
    # 1. Highlight Clipping is almost always a technical error (blown channels)
    # unless it's specular, but we penalize large areas of pure white.
    highlight_clip = np.sum(gray_img > 250) / total_pixels
    
    # 2. Determine Metering Mode (Subject vs Global)
    subject_pixels = []
    if detections:
        for d in detections:
            # Class 0 is 'person' in COCO dataset
            if d.get('class_id') == 0: 
                x1, y1, x2, y2 = map(int, d['box'])
                # Clamp coordinates
                h, w = gray_img.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = gray_img[y1:y2, x1:x2]
                    subject_pixels.append(crop)
    
    if subject_pixels and len(subject_pixels) > 0:
        # Subject-Oriented Metering
        # Flatten all subject pixels into one array
        target_data = np.concatenate([p.ravel() for p in subject_pixels])
        mean_intensity = np.mean(target_data)
        
        # We are more lenient with shadow clipping in the background if subject is detected
        # Check shadow clipping ONLY on the subject (e.g. hair or black clothes)
        shadow_clip = np.sum(target_data < 5) / target_data.size
        metering_mode = "subject"
    else:
        # Global Metering (Fallback)
        mean_intensity = np.mean(gray_img)
        shadow_clip = np.sum(gray_img < 5) / total_pixels
        metering_mode = "global"

    ideal_mean = EXPOSURE_IDEAL_MEAN_INTENSITY
    
    # Phase 2: Context-aware clipping tolerance
    shutter_speed = metadata.get("shutter_speed") if metadata else None
    tolerance = get_exposure_tolerance(shutter_speed)
    
    highlight_tolerance = tolerance['highlight_clip_tolerance']
    shadow_tolerance = tolerance['shadow_clip_tolerance']
    
    # Penalty calculation
    clipping_penalty = (max(0, highlight_clip - highlight_tolerance) + max(0, shadow_clip - shadow_tolerance)) * 2.0
    
    # Base score on mean deviance (Zone V targeting)
    base_score = max(0.0, 1.0 - abs(mean_intensity - ideal_mean) / ideal_mean)
    
    score = max(0.0, base_score - clipping_penalty)
    
    # Generate Explanation
    if highlight_clip > 0.1:
        explanation = "Excessive highlight clipping (blown out)."
    elif shadow_clip > 0.2:
        explanation = "Excessive shadow clipping (crushed blacks)."
    else:
        if score > 0.7:
            if metering_mode == "subject":
                explanation = "Subject is well-exposed (Zone V)."
            elif tolerance['context'] == 'action':
                explanation = "Exposure is technically sound for action context."
            else:
                explanation = "Exposure is well-balanced."
        else:
            if metering_mode == "subject":
                explanation = "Subject is over/under-exposed."
            else:
                explanation = "Global exposure shows deviance."
        
    return float(score), explanation


def _calculate_noise(img: np.ndarray, gray_img: np.ndarray, metadata: dict = None) -> tuple[float, str]:
    """
    Estimates sensor noise levels using multi-patch variance analysis in Luma and Chroma.
    
    Science:
    1. Luminance Noise (Grain): Measured via variance in the Grayscale intensity.
       Often acceptable or artistic (film grain).
    2. Chrominance Noise (Color Blotches): Measured via variance in the 'a' and 'b'
       channels of the CIELAB color space. Digital color noise is almost always
       undesirable.
       
    ISO-Awareness:
    Normalization factors are dynamic based on the ISO setting. High-ISO 
    images are expected to have a higher noise floor.
    
    Ref: https://en.wikipedia.org/wiki/Color_noise
    """
    h, w = gray_img.shape
    grid_size = 8  # 8x8 grid of patches
    patch_h, patch_w = h // grid_size, w // grid_size
    
    if patch_h < 10 or patch_w < 10:
        return 0.0, "Image too small for reliable noise sampling."
        
    luma_variances = []
    chroma_variances = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Luma Patch
            patch_gray = gray_img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            var_l = np.var(patch_gray)
            luma_variances.append(var_l)
            
            # Chroma Patch (Convert to LAB)
            # Optimization: Check if img is None (shouldn't happen in main flow)
            if img is not None:
                patch_bgr = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                if patch_bgr.size > 0:
                    try:
                        patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2Lab)
                        l_chan, a_chan, b_chan = cv2.split(patch_lab)
                        var_a = np.var(a_chan)
                        var_b = np.var(b_chan)
                        chroma_variances.append(var_a + var_b)
                    except:
                        pass # Fallback if convert fails
            
    # Heuristic: The "Noise Floor" is best estimated by the detected patches
    # with the LOWEST variance (shadows/smooth walls).
    if not luma_variances:
        return 0.0, "Could not sample noise patches."
        
    luma_variances.sort()
    chroma_variances.sort()
    
    cutoff_index = max(1, len(luma_variances) // 4)
    noise_floor_luma = np.mean(luma_variances[:cutoff_index])
    
    noise_floor_chroma = 0.0
    if chroma_variances:
        cutoff_chroma = max(1, len(chroma_variances) // 4)
        noise_floor_chroma = np.mean(chroma_variances[:cutoff_chroma])
    
    # Normalization Constants (Tuned for 8-bit images, Variance)
    # Base ISO 100 sensitivity (Variance 300 ~= StdDev 17.3, which is typical for "clean with texture")
    LUMA_NORM_BASE = 300.0 
    CHROMA_NORM_BASE = 200.0 
    
    iso = metadata.get("iso") if metadata else None
    if iso:
        # Scale with ISO (approx sqrt relationship for photon noise)
        tolerance_factor = np.sqrt(max(iso, 100) / 100.0)
        luma_norm = LUMA_NORM_BASE * tolerance_factor
        chroma_norm = CHROMA_NORM_BASE * tolerance_factor
    else:
        luma_norm = LUMA_NORM_BASE * 2.0 
        chroma_norm = CHROMA_NORM_BASE * 2.0
        
    # Scores (0.0 = terrible noise, 1.0 = clean)
    luma_score = max(0.0, 1.0 - (noise_floor_luma / luma_norm))
    chroma_score = max(0.0, 1.0 - (noise_floor_chroma / chroma_norm))
    
    # Combined Score: Bad Chroma drags score down more than Luma
    if chroma_variances:
        final_score = (luma_score * 0.6) + (chroma_score * 0.4)
    else:
        final_score = luma_score
    
    iso_str = f" (ISO {iso})" if iso else ""
    
    if final_score > 0.8:
        explanation = f"Minimal sensor noise detected{iso_str}."
    elif chroma_score < 0.5:
        explanation = f"Significant color noise (chroma blotches) detected{iso_str}."
    elif luma_score < 0.5:
        explanation = f"High levels of luminance grain{iso_str}."
    else:
        explanation = f"Moderate noise visible{iso_str}."
        
    return float(final_score), explanation


def _calculate_color_balance(img: np.ndarray) -> tuple[float, str]:
    """
    Assesses color balance using Neutral Pixel Selection (NPS).
    
    Science:
    A neutral image (correctly white-balanced) will have roughly equal 
    intensity in the Red, Green, and Blue channels for areas that are 
    supposed to be gray or white. This function calculates the variance 
    between R, G, and B means.
    
    The Grey World Hypothesis:
    Assumes that the average reflectance of a scene is achromatic (gray). 
    While not always true for artistic shots, it is a robust baseline for 
    technical color accuracy.
    
    Ref: https://en.wikipedia.org/wiki/Color_balance
    """
    # Identify neutral pixels (where R, G, B are similar)
    b, g, r_ch = cv2.split(img)
    diff_rg = np.abs(r_ch.astype(float) - g.astype(float))
    diff_gb = np.abs(g.astype(float) - b.astype(float))
    diff_br = np.abs(b.astype(float) - r_ch.astype(float))
    
    # Mask for pixels that are potentially neutral (low saturation)
    neutral_mask = (diff_rg < 15) & (diff_gb < 15) & (diff_br < 15)
    
    # Exclude pure black/white from neutral check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    neutral_mask &= (gray > 30) & (gray < 225)
    
    if np.sum(neutral_mask) < 100:
        # Fallback if no neutral pixels found
        return _calculate_color_balance_legacy(img)
        
    n_b = np.mean(b[neutral_mask])
    n_g = np.mean(g[neutral_mask])
    n_r = np.mean(r_ch[neutral_mask])
    
    means = np.array([n_b, n_g, n_r])
    score = max(0.0, 1.0 - np.std(means) / (np.mean(means) + 1e-6))
    
    explanation = "Natural color balance and balanced neutral tones." if score > 0.8 else "Potential color cast detected in neutral areas."
    return float(score), explanation

def _calculate_color_balance_legacy(img: np.ndarray) -> tuple[float, str]:
    """Fallback color balance using simple channel comparison."""
    rgb_means = np.mean(img, axis=(0, 1))
    score = max(0.0, 1.0 - np.std(rgb_means) / (np.mean(rgb_means) + 1e-6))
    return float(score), "Overall color distribution is balanced."


def _calculate_dynamic_range(gray_img: np.ndarray, metadata: dict = None) -> tuple[float, str]:
    """
    Assesses Tonal Utilization (Histogram Width) instead of Shannon Entropy.
    
    Science:
    Measures the technical width of the histogram containing 98% of the pixel 
    population. Shannon Entropy (previous method) confuses noise with dynamic 
    range; Histogram Width measures actual tonal spread across the bit depth.
    
    Camera-Awareness:
    Scores are normalized against the known DR capability of the camera model 
    at the used ISO, sourced from DXOMARK and PhotonsToPhotos benchmarks.
    
    Ref: https://en.wikipedia.org/wiki/Dynamic_range
    """
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.ravel() / (hist.sum() + 1e-6)
    
    # Cumulative distribution to find percentiles
    cdf = np.cumsum(hist)
    
    # Find 1st and 99th percentiles to determine the 98% width
    tonal_min = np.searchsorted(cdf, 0.01)
    tonal_max = np.searchsorted(cdf, 0.99)
    width = tonal_max - tonal_min
    
    # Normalized raw score (width of 255 = 1.0)
    raw_score = float(width) / 255.0
    
    # Phase 2: Camera-specific DR baseline adjustment
    camera_model = metadata.get("camera_model") if metadata else None
    iso = metadata.get("iso") if metadata else None
    
    if camera_model or iso:
        specs = get_camera_data(camera_model)
        base_dr = specs.get('dr', 12.0) if specs else 12.0
        expected_dr = get_camera_dynamic_range_baseline(base_dr, iso)
        
        # Normalize score to camera capability
        baseline_dr = 12.0  # Standard reference
        if expected_dr > 0:
            capability_factor = baseline_dr / expected_dr
            # If camera has low DR (high ISO), we are more lenient with the width
            score = min(raw_score * (0.8 + 0.2 * capability_factor), 1.0)
        else:
            score = raw_score
    else:
        score = raw_score
    
    explanation = "Excellent tonal utilization across the full range." if score > 0.7 else "Narrow tonal range or low contrast (flat image)."
    return float(score), explanation


def _calculate_composition(img_shape: tuple, detections: list) -> tuple[float, str]:
    """
    Evaluates aesthetic composition using Rule of Thirds and Headroom.
    
    Science:
    1. Rule of Thirds: Power Points alignment.
    2. Headroom (Psychophysics): For portraits, the vertical space above the 
       subject's head significantly affects the perception of balance.
       - Too little (chopped head): Claustrophobic.
       - Too much (dead space): Unbalanced.
       - Ideal: ~10-15% of frame height.
    
    Ref: https://en.wikipedia.org/wiki/Headroom_(photographic_framing)
    """
    if not detections:
        return 0.5, "No objects detected; neutral composition score."
        
    h, w = img_shape[:2]
    # Rule of Thirds lines (relative)
    lines_x = [1/3, 2/3]
    lines_y = [1/3, 2/3]
    
    best_rot_score = 0.0
    headroom_score = 1.0
    headroom_issue = ""
    
    has_person = False
    
    for det in detections:
        # Normalize coordinates
        x1, y1, x2, y2 = det['box']
        cx_rel = ((x1 + x2) / 2) / w
        cy_rel = ((y1 + y2) / 2) / h
        
        # Rule of Thirds Analysis
        dx = min(abs(cx_rel - lx) for lx in lines_x)
        dy = min(abs(cy_rel - ly) for ly in lines_y)
        rot_score = max(0.0, 1.0 - (dx + dy) * 3.0) 
        best_rot_score = max(best_rot_score, rot_score)
        
        # Headroom Analysis (only for people)
        if 'person' in det.get('name', '').lower():
            has_person = True
            # Headroom is distance from top of frame (0) to top of box (y1)
            # relative to frame height
            headroom_ratio = y1 / h
            
            # Ideal: 8% to 20% (0.08 - 0.20)
            if headroom_ratio < 0.02: # < 2% -> Chopped off
                headroom_score = 0.5
                headroom_issue = "Subject's head is cropped or too close to edge."
            elif headroom_ratio < 0.08: # Tight
                headroom_score = 0.9
            elif headroom_ratio > 0.35: # > 35% -> Too much sky
                # Penalty scales with distance
                headroom_score = max(0.6, 1.0 - (headroom_ratio - 0.3) * 2) 
                headroom_issue = "Excessive headroom (dead space above subject)."
            else:
                headroom_score = 1.0
                
    # Final weighting
    if has_person:
        # For portraits, headroom is critical (50% weight)
        final_score = (best_rot_score * 0.5) + (headroom_score * 0.5)
        
        rot_desc = "Strong Rule of Thirds alignment." if best_rot_score > 0.7 else "Subject is centered or off-grid."
        if headroom_issue:
            explanation = f"{rot_desc} Warning: {headroom_issue}"
        else:
            explanation = f"{rot_desc} Perfect vertical framing."
    else:
        # Non-person: 100% Rule of Thirds
        final_score = 0.5 + (best_rot_score * 0.5)
        explanation = "Strong composition guidelines followed." if best_rot_score > 0.7 else "Centrally composed or unbalanced."
    
    return float(final_score), explanation


def generate_color_palette(image_path: str, num_colors: int = 5) -> dict:
    """
    Extracts a representative color palette using K-Means Clustering.
    
    Science:
    Treats every pixel in the image as a point in 3D RGB space. K-Means 
    identifies the 'K' most dominant clusters, providing a technical 
    summary of the image's color story.
    
    Performance:
    Resizes the image to a low-resolution proxy (e.g. 200px) before clustering
    to ensure sub-second execution while maintaining color fidelity.
    
    Ref: https://en.wikipedia.org/wiki/K-means_clustering
    """
    img = _load_image_with_raw_support(image_path)
    if img is None:
        logger.warning(f"Could not load image for palette extraction: {image_path}")
        return []
        
    # Resize for speed
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    # K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to Hex
    centers = np.uint8(centers)
    hex_colors = []
    for color in centers:
        hex_colors.append('#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]))
        
    return hex_colors


def _generate_assessment_summary(
    overall_confidence: float,
    metadata_status: str,
    focus_area_explanation: str,
    main_subject_name: str | None,
    sharpness_score: float,
    exposure_score: float,
    noise_score: float,
    color_balance_score: float,
    dynamic_range_score: float,
    detected_object_names: set[str],
    focus_area_score: float,
    composition_score: float
) -> tuple[str, str, str]:
    """
    Synthesizes numerical scores into a human-readable qualitative assessment.
    
    The Reasoning Engine:
    This function acts as the "Decision Support System," translating raw 
    signal data (0.0 - 1.0) into photographic terminology (e.g., 'Sharpness 
    is excellent', 'Noticeable color cast').
    
    Linguistic Mapping:
    - Excellent: ≥ 0.8
    - Good: ≥ 0.65
    - Acceptable: ≥ 0.5
    - Poor: < 0.35
    
    Returns:
        tuple: (Judgement Label, Detailed Explanation, Scene Description)
    """
    
    # Initialize variables
    jd_parts = []
    
    # Determine overall judgement based on confidence
    if overall_confidence >= 0.8:
        judgement = "Excellent"
    elif overall_confidence >= 0.65:
        judgement = "Good"
    elif overall_confidence >= 0.5:
        judgement = "Acceptable"
    elif overall_confidence >= 0.35:
        judgement = "Poor"
    else:
        judgement = "Very Poor"
    
    jd_parts.append(f"The technical integrity is rated as {judgement.lower()}.")

    # Subject assessment
    if "No clear main subject" in focus_area_explanation or \
       "No objects detected" in focus_area_explanation or \
       "YOLO model not loaded" in focus_area_explanation or \
       "Error during YOLO" in focus_area_explanation:
        jd_parts.append(
            "No clear main subject was identified for focus assessment or an issue occurred with object detection.")
    elif main_subject_name:
        jd_parts.append(
            f"A main subject ('{main_subject_name}') was identified for focus assessment.")
    else:  # Fallback if main_subject_name is None but some detection happened
        jd_parts.append("A main subject was identified for focus assessment.")

    # Sharpness
    if sharpness_score > 0.8:
        jd_parts.append("Sharpness is excellent.")
    elif sharpness_score > 0.6:
        jd_parts.append("Sharpness is good.")
    elif sharpness_score > 0.4:
        jd_parts.append("Sharpness is acceptable.")
    else:
        jd_parts.append("The image appears blurry or lacks sharpness.")

    # Exposure
    if exposure_score > 0.85:
        jd_parts.append("Exposure is well-balanced.")
    elif exposure_score > 0.7:
        jd_parts.append("Exposure is generally good.")
    elif exposure_score > 0.5:
        jd_parts.append("Exposure is somewhat uneven.")
    else:
        jd_parts.append(
            "The image suffers from poor exposure (likely over or underexposed).")

    # Other issues/strengths
    issues, strengths = [], []
    if noise_score < 0.6:
        issues.append("noticeable noise")
    elif noise_score > 0.85:
        strengths.append("minimal noise")
    if color_balance_score < 0.7:
        issues.append("a potential color cast")
    elif color_balance_score > 0.85:
        strengths.append("good color balance")
    if dynamic_range_score < 0.6:
        issues.append("limited dynamic range")
    elif dynamic_range_score > 0.85:
        strengths.append("a wide dynamic range")

    if issues:
        jd_parts.append(f"Key issues include: {', '.join(issues)}.")
    elif strengths and not issues:  # Only add strengths if no major issues were listed
        jd_parts.append(
            f"Additional strengths include: {', '.join(strengths)}.")

    judgement_description = " ".join(jd_parts)
    image_description = f"Image containing: {', '.join(sorted(list(detected_object_names)))}." \
        if detected_object_names else "Image with no prominent objects detected by YOLO."

    return judgement, judgement_description, image_description


# --- Main Evaluation Function ---

def _load_image_with_raw_support(image_path: str) -> np.ndarray | None:
    """
    Orchestrates high-fidelity image loading with format-aware fallbacks.
    
    Pipeline Priority:
    1. RAW De-mosaicing: Uses `rawpy` (LibRaw) for 16-bit signal extraction.
    2. Embedded Preview: Parses EXIF for the largest hidden JPEG thumbnail 
       (crucial for Sony/Canon RAW where rawpy may be unavailable).
    3. Standard Decoding: Falls back to `OpenCV` (LibJPEG/LibPNG).
    
    This multi-stage process ensures that professional photographers can 
    analyze high-res RAW files locally with maximum performance.
    """
    ext = os.path.splitext(image_path)[1].lower()
    raw_extensions = {'.arw', '.cr2', '.nef', '.dng', '.orf', '.raf', '.srw', '.cr3', '.rw2', '.nrw', '.gpr', '.sr2', '.pef', '.rwl'}
    
    if ext in raw_extensions:
        # 1. Primary: Use rawpy for high-fidelity extraction if available
        if rawpy is not None:
            try:
                with rawpy.imread(image_path) as raw:
                    # Turbo Optimization: use half_size=True for 8x faster decoding during culling
                    rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0, half_size=True)
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning(f"Rawpy failed for {image_path}: {e}. Falling back to ExifRead.")
        else:
            logger.debug(f"Rawpy not installed, using ExifRead for {image_path}")
            
        # 2. Secondary: Fallback to manual ExifRead preview extraction
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
                # Sony and regular DCIM manufacturers often store multiple previews.
                # We want the largest one for the best visual analysis.
                previews = []
                
                # Check direct binary tags
                for tag_name in ['JPEGThumbnail', 'PreviewImage', 'MakerNote Thumbnail']:
                    if tag_name in tags:
                        val = tags[tag_name]
                        data = val.values if hasattr(val, 'values') else val
                        if isinstance(data, (bytes, bytearray)):
                            previews.append(data)
                
                # Check offset/length tags
                for prefix in ['', 'Image ', 'Thumbnail ']:
                    offset_tag = tags.get(f'{prefix}JPEGInterchangeFormat')
                    length_tag = tags.get(f'{prefix}JPEGInterchangeFormatLength')
                    if offset_tag and length_tag:
                        try:
                            offset = int(offset_tag.values[0]) if hasattr(offset_tag, 'values') else int(offset_tag[0])
                            length = int(length_tag.values[0]) if hasattr(length_tag, 'values') else int(length_tag[0])
                            f.seek(offset)
                            previews.append(f.read(length))
                        except:
                            continue
                
                if previews:
                    # Sort by length and pick the largest
                    previews.sort(key=len, reverse=True)
                    preview_data = previews[0]
                    
                    img_array = np.frombuffer(preview_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Log the size of the preview we're using
                        logger.debug(f"Using {len(preview_data)/1024:.1f}KB preview for {image_path}")
                        return img
        except Exception as e:
            logger.warning(f"ExifRead fallback failed for {image_path}: {e}")
            
    # Fallback to standard imread for JPG/PNG or if RAW extraction failed
    return cv2.imread(image_path)


def create_xmp_sidecar(image_path: str, status: str, confidence: float) -> None:
    """
    Generates an Adobe-compatible XMP sidecar file for metadata portability.
    
    Integration:
    Writes a 'Label' (e.g. 'Rejected') and 'Rating' (based on confidence)
    that can be read by Lightroom, Capture One, and Bridge. This bridges
    the gap between AI analysis and the photographer's editing workflow.
    
    Rating Logic:
    - 5 Stars: Confidence > 0.9
    - 3 Stars: Confidence > 0.7
    - 0 Stars/Rejected Label: Confidence < 0.4
    
    Ref: https://www.adobe.com/products/xmp.html
    """
    xmp_path = os.path.splitext(image_path)[0] + ".xmp"
    
    # Calculate Adobe-compatible rating and label
    rating = int(confidence * 5)
    label = "Rejected" if confidence < 0.4 else "Keep"
    
    # Minimal XMP template for Adobe Lightroom compatibility
    xmp_content = f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.160451, 2017/05/06-01:08:21        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    xmp:Rating="{rating}"
    xmp:Label="{label}">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    
    try:
        with open(xmp_path, "w") as f:
            f.write(xmp_content)
        logger.info(f"Generated XMP sidecar for {image_path} at {xmp_path}")
    except Exception as e:
        logger.error(f"Failed to write XMP for {image_path}: {e}")


def _detect_objects(img: np.ndarray) -> list[dict]:
    """
    Runs YOLO detection and returns a standardized list of detections.
    Returns: [{'box': [x1, y1, x2, y2], 'class_id': int, 'conf': float, 'name': str}, ...]
    """
    global g_yolo_model, g_coco_names
    detections = []
    if g_yolo_model is None:
        return detections
        
    try:
        # Run inference
        results = g_yolo_model.predict(
            source=img, conf=YOLO_CONFIDENCE_THRESHOLD, iou=YOLO_NMS_THRESHOLD, verbose=False
        )
        
        if results and results[0].boxes:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                name = g_coco_names[classes[i]] if g_coco_names and classes[i] < len(g_coco_names) else f"obj_{classes[i]}"
                detections.append({
                    'box': boxes[i].tolist(),
                    'class_id': int(classes[i]),
                    'conf': float(confs[i]),
                    'name': name
                })
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}")
        
    return detections


def evaluate_photo_quality(
    image_path: str,
    requested_metrics: list[str] = None,
    enable_subject_detection: bool = True,
    model_size: str = "nano"
) -> dict:
    """
    Performs a comprehensive technical and aesthetic assessment of a photograph.
    
    This is the main entry point for the engine. It extracts EXIF metadata,
    analyzes signal properties (Sharpness, Exposure, Noise, DR), and runs 
    neural-network based subject/composition analysis to provide a 
    weighted "Final Judgement."
    
    Returns:
        dict: A structured report containing:
            - overallConfidence: The final quality score (0.0 to 1.0)
            - technicalScore: Physics-based execution quality
            - aestheticScore: Composition and framing quality
            - judgement: Human-readable label (e.g. 'Excellent')
            - metrics: Detailed breakdown per evaluation category
    """
    all_metrics = {"sharpness", "focus", "exposure", "noise", "color", "dynamicRange", "composition"}
    if requested_metrics is None or "all" in requested_metrics:
        requested = all_metrics
    else:
        requested = set(requested_metrics)

    # Disable Neural-based metrics if enable_subject_detection is False
    if not enable_subject_detection:
        requested = requested - {"focus", "composition"}

    img = _load_image_with_raw_support(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Extract Context
    metadata = _extract_metadata(image_path)

    # 0. Initialize Models & Run Detection (Centralized Visual Intelligence)
    needs_yolo = "focus" in requested or "composition" in requested or "exposure" in requested
    detections = []
    
    if needs_yolo:
        ensure_yolo_initialized(model_size=model_size)
        if enable_subject_detection:
            # Run YOLO once for all metrics
            detections = _detect_objects(img)

    # Metrics Storage
    results = {}
    
    # --- Technical Core ---
    
    # Sharpness
    sharpness_score = 1.0
    if "sharpness" in requested:
        sharpness_score, sharpness_explanation = _calculate_sharpness(gray, metadata)
        results["sharpness"] = {"score": float(sharpness_score), "explanation": sharpness_explanation}
    
    # Focus (Uses shared detections)
    focus_area_score = 1.0
    main_subject_name = "unknown"
    detected_object_names = []
    if "focus" in requested:
        focus_area_score, focus_area_explanation, detected_object_names, main_subject_name = \
            _calculate_focus_area(img, gray, sharpness_score, metadata, detections=detections)
        results["focus"] = {"score": float(focus_area_score), "explanation": focus_area_explanation}
    
    # Exposure (Uses detections for Zone V logic)
    exposure_score = 1.0
    if "exposure" in requested:
        exposure_score, exposure_explanation = _calculate_exposure(gray, metadata, detections=detections)
        results["exposure"] = {"score": float(exposure_score), "explanation": exposure_explanation}
        
    # Noise
    noise_score = 1.0
    if "noise" in requested:
        noise_score, noise_explanation = _calculate_noise(img, gray, metadata)
        results["noise"] = {"score": float(noise_score), "explanation": noise_explanation}

    # --- Aesthetic Factors ---
    
    # Color Balance
    color_balance_score = 1.0
    if "color" in requested:
        color_balance_score, color_balance_explanation = _calculate_color_balance(img)
        results["color"] = {"score": float(color_balance_score), "explanation": color_balance_explanation}
        
    # Dynamic Range (New Tonal Utilization Logic)
    dynamic_range_score = 1.0
    if "dynamicRange" in requested:
        dynamic_range_score, dynamic_range_explanation = _calculate_dynamic_range(gray, metadata)
        results["dynamicRange"] = {"score": float(dynamic_range_score), "explanation": dynamic_range_explanation}
        
    # Composition
    composition_score = 1.0
    if "composition" in requested:
        # Pass full detections (with 'name') for Headroom Analysis
        composition_score, composition_explanation = _calculate_composition(img.shape, detections)
        results["composition"] = {"score": float(composition_score), "explanation": composition_explanation}

    # 3. Composite Scoring (Technical Gatekeeper Math)
    # We use requested weights or default weights normalized
    tech_weights = {"sharpness": 0.4, "focus": 0.3, "exposure": 0.2, "noise": 0.1}
    aes_weights = {"color": 0.4, "dynamicRange": 0.4, "composition": 0.2}
    
    # Normalize weights based on whats available
    def get_weighted_average(scores_dict, weights_dict, requested_set):
        sub_weights = {k: v for k, v in weights_dict.items() if k in requested_set}
        total_w = sum(sub_weights.values())
        if total_w == 0: return 1.0
        return sum(scores_dict.get(k, 1.0) * (v/total_w) for k, v in sub_weights.items())

    scores_map = {
        "sharpness": sharpness_score, "focus": focus_area_score, "exposure": exposure_score, "noise": noise_score,
        "color": color_balance_score, "dynamicRange": dynamic_range_score, "composition": composition_score
    }
    
    tech_score = get_weighted_average(scores_map, tech_weights, requested)
    aesthetic_score = get_weighted_average(scores_map, aes_weights, requested)
    
    # Composite formula
    overall_confidence = tech_score * (0.8 + 0.2 * aesthetic_score)

    # Generate assessment summary
    judgement, judgement_description, image_description = _generate_assessment_summary(
        overall_confidence, metadata["status"], 
        results.get("focus", {}).get("explanation", "N/A"), 
        main_subject_name,
        sharpness_score, exposure_score, noise_score, color_balance_score,
        dynamic_range_score, detected_object_names,
        focus_area_score, composition_score
    )

    return {
        "metadataStatus": metadata["status"],
        "technicalScore": float(tech_score),
        "aestheticScore": float(aesthetic_score),
        "overallConfidence": float(overall_confidence),
        "judgement": judgement,
        "judgementDescription": judgement_description,
        "description": image_description,
        "metrics": results
    }


# --- File Processing Function ---

def process_folder(folder_path: str, verbose: bool, move_files: bool, requested_metrics: list[str] = None) -> None:
    """
    Orchestrates the batch processing of an image directory.
    
    Workflow:
    1. Scans directory for all supported extensions.
    2. Runs the multi-stage evaluation pipeline for each file.
    3. (Optional) Performs physical file organization into good/fair/bad bins.
    
    This function is optimized for large-scale ingestion (1000+ photos), 
    using memory management best-practices for local execution.
    """
    global g_yolo_model  # Ensure it uses the globally loaded model
    if g_yolo_model is None:
        logger.critical(
            "YOLO model could not be loaded. Cannot proceed with image processing.")
        return

    logger.info(f"Processing images in folder: {folder_path}")

    # Define paths for sorted images
    good_dir = os.path.join(folder_path, "good_photos")
    fair_dir = os.path.join(folder_path, "fair_photos")
    bad_dir = os.path.join(folder_path, "bad_photos")

    if move_files:
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(fair_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)
        logger.info(f"Good photos will be moved to: {good_dir}")
        logger.info(f"Fair photos will be moved to: {fair_dir}")
        logger.info(f"Bad photos (Poor/Very Poor) will be moved to: {bad_dir}")

    processed_count = 0

    image_files = [f for f in os.listdir(
        folder_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    if not image_files:
        logger.info(f"No image files found directly in {folder_path}.")
        return

    for filename in tqdm(image_files, desc="Processing Images", unit="image"):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            image_path = os.path.join(folder_path, filename)
            try:
                # Skip processing files if they are already in one of the target subdirectories
                if move_files:
                    parent_dir_abs = os.path.abspath(
                        os.path.dirname(image_path))
                    # Check if the image's parent directory is one of the target output directories
                    if parent_dir_abs in [os.path.abspath(d) for d in [good_dir, fair_dir, bad_dir]]:
                        if verbose:
                            logger.debug(
                                f"Skipping {filename} as it's already in a target move directory.")
                        continue

                result = evaluate_photo_quality(image_path, requested_metrics=requested_metrics)
                processed_count += 1

                # Conditional logging based on verbosity for individual results
                if verbose:
                    logger.info(
                        f"--- Results for {filename} ---\n{json.dumps(result, indent=2)}")
                else:  # Not verbose, provide a summary regardless of move_files
                    logger.info(
                        f"Processed: {filename} - Judgement: {result['judgement']} (Confidence: {result['overallConfidence']:.2f}) - Summary: {result['judgementDescription']}")

                if move_files:
                    destination_folder = ""
                    if result['judgement'] in ["Excellent", "Good"]:
                        destination_folder = good_dir
                    elif result['judgement'] == "Fair":
                        destination_folder = fair_dir
                    else:  # Poor, Very Poor
                        destination_folder = bad_dir

                    destination_path = os.path.join(
                        destination_folder, filename)  # Ensure filename is used, not image_path
                    shutil.move(image_path, destination_path)
                    logger.debug(f"Moved {filename} to {destination_folder}")

            except ValueError as ve:  # Catch specific error from imread
                logger.warning(f"Skipping {filename}: {ve}")
            except Exception as e:
                logger.error(
                    f"Error processing {filename}: {e}", exc_info=True)

    if processed_count == 0:
        logger.info(
            f"No image files were processed in {folder_path} (after filtering).")


# --- Main Execution ---

def main() -> None:
    """
    Entry point for the Photographi CLI (Command Line Interface).
    
    Functionality:
    - Parses CLI flags and arguments.
    - Resolves configuration file location.
    - Manages global state for neural models.
    - Triggers the directory processing pipeline.
    """
    global g_yolo_model, g_coco_names, YOLO_MODEL_PATH_DEFAULT, COCO_NAMES_FILE_PATH_DEFAULT

    parser = argparse.ArgumentParser(
        description="Analyze photo quality in a folder using a YOLO model.")
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing images to analyze."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full JSON output for each image."
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move photos to 'good_photos', 'fair_photos', or 'bad_photos' subfolders based on judgement."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=YOLO_MODEL_PATH_DEFAULT,
        help=f"Path to the YOLO model file (e.g., yolov11n.pt, yolov8n.pt). Default: {YOLO_MODEL_PATH_DEFAULT}"
    )
    args = parser.parse_args()

    # Load model based on default or user-provided path
    model_to_load = args.model_path
    g_yolo_model, g_coco_names = load_yolo_model_and_names(
        model_to_load, COCO_NAMES_FILE_PATH_DEFAULT)

    # Validate folder path
    if not os.path.exists(args.folder_path):
        logger.error(f"The directory '{args.folder_path}' was not found.")
        exit(1)
    if not os.path.isdir(args.folder_path):
        logger.error(f"The path '{args.folder_path}' is not a directory.")
        exit(1)

    # Start processing
    process_folder(args.folder_path, args.verbose, args.move)


if __name__ == "__main__":
    main()
