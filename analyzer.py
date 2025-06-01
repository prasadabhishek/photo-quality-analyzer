# --- Dependency Check ---
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from tqdm import tqdm
except ImportError as e:
    print(f"ImportError: {e}")
    print("One or more required Python packages are not installed.")
    print("Please install the necessary dependencies by running:")
    print("pip install -r requirements.txt")
    print("If you don't have 'requirements.txt', ensure you have opencv-python, numpy, and ultralytics installed.")
    exit(1)

# --- Standard Library Imports ---
import json
import os
import logging
import argparse
import shutil  # Added for moving files
import configparser

# Note: YOLO is imported in the try-except block above

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'

if not os.path.exists(CONFIG_FILE_PATH):
    logger.critical(
        f"Configuration file '{CONFIG_FILE_PATH}' not found. Please create it.")
    logger.critical(
        "You can use the example provided in the README or documentation.")
    exit(1)

try:
    config.read(CONFIG_FILE_PATH)

    SHARPNESS_NORMALIZATION_FACTOR = config.getfloat(
        'NormalizationFactors', 'sharpness', fallback=1000.0)
    FOCUS_AREA_NORMALIZATION_FACTOR = config.getfloat(
        'NormalizationFactors', 'focus_area', fallback=1000.0)
    NOISE_NORMALIZATION_FACTOR = config.getfloat(
        'NormalizationFactors', 'noise', fallback=50.0)

    EXPOSURE_IDEAL_MEAN_INTENSITY = config.getfloat(
        'Thresholds', 'exposure_ideal_mean', fallback=128.0)
    DYNAMIC_RANGE_MAX_VALUE = config.getfloat(
        'Thresholds', 'dynamic_range_max', fallback=255.0)
    YOLO_CONFIDENCE_THRESHOLD = config.getfloat(
        'Thresholds', 'yolo_confidence', fallback=0.5)
    YOLO_NMS_THRESHOLD = config.getfloat(
        'Thresholds', 'yolo_nms', fallback=0.45)

    OVERALL_CONF_TECH_WEIGHT = config.getfloat(
        'Weights', 'overall_tech', fallback=0.6)
    OVERALL_CONF_OTHER_WEIGHT = config.getfloat(
        'Weights', 'overall_other', fallback=0.4)

    YOLO_MODEL_PATH_DEFAULT = config.get(
        'Models', 'default_yolo_model', fallback="yolo12x.pt")

    JUDGEMENT_EXCELLENT = config.getfloat(
        'JudgementLevels', 'excellent', fallback=0.9)
    JUDGEMENT_GOOD = config.getfloat('JudgementLevels', 'good', fallback=0.7)
    JUDGEMENT_FAIR = config.getfloat('JudgementLevels', 'fair', fallback=0.5)
    JUDGEMENT_POOR = config.getfloat('JudgementLevels', 'poor', fallback=0.3)

except (configparser.Error, ValueError) as e:
    logger.critical(
        f"Error reading configuration file '{CONFIG_FILE_PATH}': {e}")
    exit(1)

COCO_NAMES_FILE_PATH_DEFAULT = "coco.names"

# Global variables for the loaded model and names
g_yolo_model = None
g_coco_names = None


def load_yolo_model_and_names(model_path: str, coco_names_file_path: str) -> tuple[YOLO | None, list[str] | None]:
    """
    Loads the YOLO model and class names.
    Tries to load class names from the model first, then falls back to a .names file.
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


# --- Metric Calculation Helper Functions ---

def _calculate_sharpness(gray_img: np.ndarray) -> tuple[float, str]:
    """Calculates overall image sharpness using Laplacian variance."""
    laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    score = min(laplacian_var / SHARPNESS_NORMALIZATION_FACTOR, 1.0)
    explanation = "Edges are sharp with high variance." if score > 0.8 else "Edges are slightly blurry."
    return score, explanation


def _calculate_focus_area(
    img: np.ndarray, gray_img: np.ndarray, overall_sharpness_score: float
) -> tuple[float, str, set[str], str | None]:
    """Calculates focus on the main subject using YOLO."""
    global g_yolo_model, g_coco_names  # Access global model and names

    height, width = gray_img.shape
    focus_score = overall_sharpness_score  # Default if no subject or error
    focus_explanation = "No main subject detected for focus; using overall sharpness."
    detected_obj_names: set[str] = set()
    main_subj_name: str | None = None

    if g_yolo_model is not None:
        try:
            yolo_results = g_yolo_model.predict(
                source=img, conf=YOLO_CONFIDENCE_THRESHOLD, iou=YOLO_NMS_THRESHOLD, verbose=False
            )
            if yolo_results and yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
                # Results for the first (and only) image
                result = yolo_results[0]
                boxes_data = result.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
                confidences_data = result.boxes.conf.cpu().numpy()
                class_ids_data = result.boxes.cls.cpu().numpy().astype(int)

                # Find the detection with the highest confidence to consider as main subject
                best_detection_idx = np.argmax(confidences_data)

                x1_main, y1_main, x2_main, y2_main = boxes_data[best_detection_idx]
                main_subject_class_id = class_ids_data[best_detection_idx]

                # Define Region of Interest (ROI) for the main subject
                roi_x1, roi_y1 = int(max(0, x1_main)), int(max(0, y1_main))
                roi_x2, roi_y2 = int(min(x2_main, width)), int(
                    min(y2_main, height))

                if roi_x2 > roi_x1 and roi_y2 > roi_y1:  # Ensure valid ROI dimensions
                    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi.size > 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        laplacian_var_roi = cv2.Laplacian(
                            gray_roi, cv2.CV_64F).var()
                        focus_score = min(
                            laplacian_var_roi / FOCUS_AREA_NORMALIZATION_FACTOR, 1.0)
                        focus_explanation = "Main subject is in sharp focus." if focus_score > 0.8 \
                                            else "Main subject is slightly out of focus."
                        # Get the name of the main subject
                        if g_coco_names and main_subject_class_id < len(g_coco_names) and g_coco_names[main_subject_class_id] is not None:
                            main_subj_name = g_coco_names[main_subject_class_id]
                        else:  # Fallback if name not found
                            main_subj_name = f"object_id_{main_subject_class_id}"
                    else:
                        focus_explanation = "Invalid ROI (empty); using overall sharpness."
                else:
                    focus_explanation = "Invalid ROI (zero area); using overall sharpness."

                # Collect names of all detected objects
                for i in range(len(class_ids_data)):
                    class_id = class_ids_data[i]
                    if g_coco_names and class_id < len(g_coco_names) and g_coco_names[class_id] is not None:
                        detected_obj_names.add(g_coco_names[class_id])
                    else:  # Fallback if name not found
                        detected_obj_names.add(f"object_id_{class_id}")
            else:  # No YOLO results or no boxes detected
                focus_explanation = "No objects detected by YOLO; using overall sharpness."
        except Exception as e_yolo:
            logger.error(
                f"Error during YOLO prediction or processing: {e_yolo}", exc_info=True)
            focus_explanation = "Error during YOLO processing; using overall sharpness."
    else:  # g_yolo_model is None
        focus_explanation = "YOLO model not loaded; using overall sharpness."
    return focus_score, focus_explanation, detected_obj_names, main_subj_name


def _calculate_exposure(gray_img: np.ndarray) -> tuple[float, str]:
    """Calculates image exposure based on mean intensity."""
    mean_intensity = np.mean(gray_img)
    score = max(0.0, 1.0 - abs(mean_intensity -
                EXPOSURE_IDEAL_MEAN_INTENSITY) / EXPOSURE_IDEAL_MEAN_INTENSITY)
    explanation = "Brightness is balanced with details in shadows and highlights." if score > 0.8 \
                  else "Image is slightly over/underexposed."
    return score, explanation


def _calculate_noise(gray_img: np.ndarray) -> tuple[float, str]:
    """Estimates image noise from a sample region."""
    # Using a small top-left region; for more robustness, consider analyzing multiple patches
    # or using more advanced noise estimation techniques if this proves insufficient.
    h, w = gray_img.shape
    roi_h, roi_w = min(50, h), min(50, w)  # Ensure ROI is within image bounds
    if roi_h == 0 or roi_w == 0:  # Handle very small images
        return 0.0, "Image too small to reliably estimate noise."

    noise_region = gray_img[0:roi_h, 0:roi_w]
    noise_level = np.std(noise_region)
    score = max(1.0 - noise_level / NOISE_NORMALIZATION_FACTOR, 0.0)
    explanation = "Minimal noise detected." if score > 0.8 else "Noticeable graininess present."
    return score, explanation


def _calculate_color_balance(img: np.ndarray) -> tuple[float, str]:
    """Assesses color balance by comparing mean values of color channels."""
    rgb_means = np.mean(img, axis=(0, 1))  # Mean of B, G, R channels
    # Score is higher if the standard deviation of channel means is low relative to their average
    score = max(0.0, 1.0 - np.std(rgb_means) / (np.mean(rgb_means) +
                1e-6)) if np.mean(rgb_means) > 1e-6 else 0.0
    explanation = "Colors are natural and balanced." if score > 0.8 else "Slight color cast detected."
    return score, explanation


def _calculate_dynamic_range(gray_img: np.ndarray) -> tuple[float, str]:
    """Calculates dynamic range from the histogram of the grayscale image."""
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # Indices of bins with non-zero counts
    active_bins = np.where(hist[:, 0] > 0)[0]
    if len(active_bins) > 0:
        # Difference between highest and lowest active bin
        dynamic_range_val = active_bins[-1] - active_bins[0]
        score = min(float(dynamic_range_val) / DYNAMIC_RANGE_MAX_VALUE, 1.0)
    else:  # Flat image (e.g., all black or all white)
        score = 0.0
    explanation = "Wide dynamic range with details in all tones." if score > 0.8 \
                  else "Limited dynamic range with some detail loss."
    return score, explanation


def _generate_assessment_summary(
    overall_confidence: float,
    focus_area_explanation: str,
    main_subject_name: str | None,
    sharpness_score: float,
    exposure_score: float,
    noise_score: float,
    color_balance_score: float,
    dynamic_range_score: float,
    detected_object_names: set[str]
) -> tuple[str, str, str]:
    """Generates judgement, judgement description, and image description based on scores."""
    judgement = (
        "Excellent" if overall_confidence >= JUDGEMENT_EXCELLENT else
        "Good" if overall_confidence >= JUDGEMENT_GOOD else
        "Fair" if overall_confidence >= JUDGEMENT_FAIR else
        "Poor" if overall_confidence >= JUDGEMENT_POOR else
        "Very Poor"
    )

    jd_parts = []
    # Overall quality statement
    if overall_confidence >= 0.9:
        jd_parts.append("Overall technical quality is excellent.")
    elif overall_confidence >= 0.7:
        jd_parts.append("Overall technical quality is good.")
    elif overall_confidence >= 0.5:
        jd_parts.append("Overall technical quality is fair.")
    else:
        jd_parts.append("Overall technical quality is poor.")

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

def evaluate_photo_quality(image_path: str) -> dict:
    """
    Orchestrates the evaluation of a photograph's quality by calling helper functions
    for each metric and then summarizing the results.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness (overall)
    sharpness_score, sharpness_explanation = _calculate_sharpness(gray)

    # 2. Focus Area (using YOLO model)
    focus_area_score, focus_area_explanation, detected_object_names, main_subject_name = \
        _calculate_focus_area(img, gray, sharpness_score)

    # 3. Exposure
    exposure_score, exposure_explanation = _calculate_exposure(gray)

    # 4. Noise
    noise_score, noise_explanation = _calculate_noise(gray)

    # 5. Color Balance
    color_balance_score, color_balance_explanation = _calculate_color_balance(
        img)

    # 6. Dynamic Range
    dynamic_range_score, dynamic_range_explanation = _calculate_dynamic_range(
        gray)

    # Calculate overall confidence
    tech_scores = [sharpness_score, focus_area_score,
                   exposure_score, noise_score]
    other_scores = [color_balance_score, dynamic_range_score]
    avg_tech_score = sum(tech_scores) / \
        len(tech_scores) if tech_scores else 0.0
    avg_other_score = sum(other_scores) / \
        len(other_scores) if other_scores else 0.0
    overall_confidence = (avg_tech_score * OVERALL_CONF_TECH_WEIGHT) + \
                         (avg_other_score * OVERALL_CONF_OTHER_WEIGHT)

    # Generate assessment summary
    judgement, judgement_description, image_description = _generate_assessment_summary(
        overall_confidence, focus_area_explanation, main_subject_name,
        sharpness_score, exposure_score, noise_score, color_balance_score,
        dynamic_range_score, detected_object_names
    )

    return {
        "Sharpness": {"confidence": float(sharpness_score), "explanation": sharpness_explanation},
        "Focus Area": {"confidence": float(focus_area_score), "explanation": focus_area_explanation},
        "Exposure": {"confidence": float(exposure_score), "explanation": exposure_explanation},
        "Noise": {"confidence": float(noise_score), "explanation": noise_explanation},
        "Color Balance": {"confidence": float(color_balance_score), "explanation": color_balance_explanation},
        "Dynamic Range": {"confidence": float(dynamic_range_score), "explanation": dynamic_range_explanation},
        "description": image_description,
        "overall_confidence": float(overall_confidence),
        "judgement_description": judgement_description,
        "judgement": judgement
    }


# --- File Processing Function ---

def process_folder(folder_path: str, verbose: bool, move_files: bool):
    """
    Process all images in a folder and print quality evaluation results to the terminal.
    Optionally moves files to 'good', 'fair', or 'bad' subdirectories based on judgement.
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
        folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        logger.info(f"No image files found directly in {folder_path}.")
        return

    for filename in tqdm(image_files, desc="Processing Images", unit="image"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
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

                result = evaluate_photo_quality(image_path)
                processed_count += 1

                # Conditional logging based on verbosity for individual results
                if verbose:  # Full JSON output if verbose
                    logger.info(
                        f"--- Results for {filename} ---\n{json.dumps(result, indent=2)}")
                elif not move_files:  # Not verbose AND not moving files, provide a summary
                    logger.info(
                        f"Processed: {filename} - Judgement: {result['judgement']} (Confidence: {result['overall_confidence']:.2f}) - Summary: {result['judgement_description']}")

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

def main():
    """
    Parses command-line arguments, initializes the model, and starts image processing.
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
