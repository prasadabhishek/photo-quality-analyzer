# Photo Quality Analyzer 📸✨

Are you drowning in a sea of digital photos, unsure which ones truly shine? The **Photo Quality Analyzer** is your intelligent command-line companion, designed to meticulously evaluate your photographs based on a suite of technical metrics. Think of it as an AI-powered first pass, giving you objective insights into the technical craftsmanship of your images.

Leveraging the power of **OpenCV** for image processing and a **YOLO model** (via the `ultralytics` library, defaulting to the model specified in `config.ini`, e.g., `yolo12x.pt`) for intelligent object detection, this script goes beyond simple checks. It identifies main subjects to assess focus specifically on what matters, giving you actionable insights.

Whether you're a photographer aiming to refine your technique, a content creator curating your best visuals, or a developer needing to automate image quality assessments, this tool offers a data-driven perspective.

## Features

The Photo Quality Analyzer dissects your images, providing scores and explanations for:

*   🎯 **Intelligent Focus Assessment**:
    *   Utilizes a YOLO model (defaulting to the model in `config.ini`, e.g., `yolo12x.pt`) to detect the primary subject.
    *   Calculates sharpness specifically on this main subject, not just the overall image.
*   🔪 **Sharpness Analysis**: Measures overall image acutance and edge clarity.
*   💡 **Exposure Evaluation**: Determines if the image is well-exposed, under-exposed, or over-exposed.
*   🔇 **Noise Estimation**: Quantifies the level of visual noise or grain.
*   🎨 **Color Balance Check**: Assesses color neutrality and detects potential color casts.
*   🌈 **Dynamic Range Assessment**: Evaluates the tonal range from the darkest shadows to the brightest highlights.

**Output & Functionality:**

*   📊 **Detailed Metrics**: For each analyzed aspect, get a `confidence` score (0.0-1.0) and a clear `explanation`.
*   🏆 **Overall Judgement**: Receives an `overall_confidence` score and a qualitative `judgement` (e.g., "Excellent", "Good", "Fair", "Poor", "Very Poor") based on thresholds in `config.ini`.
*   📝 **Summarized Insights**:
    *   `judgement_description`: A human-readable summary of the findings.
    *   `image_description`: A list of objects detected by the YOLO model.
*   📁 **Automated Organization**: Optionally move images into `good_photos`, `fair_photos`, and `bad_photos` subfolders based on their scores.
*   🗣️ **Verbose Output**: Get the full JSON output for in-depth analysis if needed, logged to the console.
*   🔧 **Flexible Model Choice**: Specify a custom path to any compatible YOLO `.pt` model file via command-line argument.
*   ⚙️ **Configurable Parameters**: Fine-tune analysis thresholds, weights, and model paths via an external `config.ini` file.
*   ⏳ **Progress Bar**: See a progress bar (thanks to `tqdm`) when processing multiple images.

## How It Works & What to Expect

This script provides a **technical, best-guess estimate** of image quality based on established image processing algorithms and AI-driven object detection. It's a tool to guide and inform, not an absolute arbiter of artistic merit.

**Metric Calculation Highlights:**

*   **Sharpness & Focus Area:**
    *   Calculated using the variance of the Laplacian operator on the grayscale image. Higher variance generally indicates sharper edges.
    *   For "Focus Area," this calculation is applied specifically to the region of interest (ROI) of the main subject detected by the YOLO model. If no subject is detected, it defaults to the overall image sharpness.
*   **Exposure:**
    *   Assessed by analyzing the mean intensity of the grayscale image. The score reflects how close the mean intensity is to an ideal mid-gray value (configurable in `config.ini`).
*   **Noise:**
    *   Estimated by calculating the standard deviation of pixel intensities in a sample region of the image (typically a flat area, though currently a fixed top-left region). Higher standard deviation in such an area can indicate more noise.
*   **Color Balance:**
    *   Evaluated by comparing the mean values of the Blue, Green, and Red color channels. A lower standard deviation among these means suggests better color neutrality.
*   **Dynamic Range:**
    *   Estimated from the image's histogram by looking at the spread of active pixel intensity values. A wider spread suggests a better dynamic range.

**Overall Confidence & Judgement:**

1.  **Individual Scores:** Each metric above receives a confidence score (0.0 to 1.0).
2.  **Categorization:** These scores are grouped into "technical" (Sharpness, Focus Area, Exposure, Noise) and "other" (Color Balance, Dynamic Range) categories.
3.  **Averaging:** The scores within each category are averaged.
4.  **Weighted Sum:** The `overall_confidence` is a weighted sum of these category averages. The weights (`overall_tech` and `overall_other`) are defined in `config.ini`.
    ```
    overall_confidence = (avg_tech_score * weight_tech) + (avg_other_score * weight_other)
    ```
5.  **Judgement:** The `overall_confidence` score is then mapped to a qualitative `judgement` (Excellent, Good, Fair, Poor, Very Poor) based on thresholds defined in the `[JudgementLevels]` section of `config.ini`.

**User Expectations:**

*   **Technical Guidance:** Expect the script to provide a good technical baseline. It can quickly highlight images that are technically problematic (e.g., very blurry, poorly exposed, extremely noisy).
*   **Not an Art Critic:** This tool does **not** evaluate artistic composition, emotional impact, storytelling, or subjective beauty. A technically "Excellent" photo might not be artistically compelling, and vice-versa.
*   **Configuration Matters:** The results, especially the final `judgement`, are influenced by the thresholds and weights in `config.ini`. Feel free to experiment with these values to align the script's output with your specific needs or preferences.
*   **YOLO Model Dependency:** The quality of object detection (and thus the "Focus Area" assessment and "image_description") depends heavily on the chosen YOLO model and its training.
*   **Best Effort:** The calculations are estimates. For instance, noise estimation is based on a sample region, and focus is based on Laplacian variance, which is a common but not infallible method.

Use this script as one of several tools in your image assessment workflow. It excels at flagging technical issues and providing a consistent baseline for large batches of photos.

## Usage

To run the Photo Quality Analyzer, navigate to its directory in your terminal and use the following command structure:

```bash
python analyzer.py --folder_path /path/to/your/images [options]
```

**Command-Line Arguments:**

*   `--folder_path <path>`: **(Required)** Path to the folder containing the images to analyze.
*   `--model_path <model.pt>`: (Optional) Path to a specific YOLO model file (e.g., `yolov8n.pt`).
    *   Overrides the default model specified in `config.ini`.
    *   Standard Ultralytics model names (like `yolov8n.pt`) will be downloaded automatically on first use if not found locally (requires internet).
*   `--verbose`: (Optional) Enables more detailed logging, including the full JSON output for each analyzed image to the console.
*   `--move`: (Optional) Move photos to `good_photos`, `fair_photos`, or `bad_photos` subfolders based on their judgement.

## Setup

Get up and running in a few simple steps:

1.  **Prerequisites:**
    *   Python 3.7 or higher.

2.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd photo-quality-analyzer # Or your repository's directory name
    ```

3.  **(Recommended) Create and Activate a Virtual Environment:**
    This keeps your project dependencies isolated.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # For Windows (Command Prompt): .venv\Scripts\activate.bat
    # For Windows (PowerShell): .venv\Scripts\Activate.ps1
    ```

4.  **Install Dependencies:**
    With the virtual environment activated, install all the necessary Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configuration File (`config.ini`):**
    *   A `config.ini` file **must** exist in the same directory as `photo_analyzer.py`.
    *   A default `config.ini` is provided with the project. You can modify this file to fine-tune the analysis parameters.
    *   If this file is missing, the script will exit with an error. Ensure it's present in the script's directory.
    *   The structure of `config.ini` (and its default values) is as follows:
        ```ini
        [NormalizationFactors]
        sharpness = 1000.0
        focus_area = 1000.0
        noise = 50.0

        [Thresholds]
        exposure_ideal_mean = 128.0
        dynamic_range_max = 255.0
        yolo_confidence = 0.5
        yolo_nms = 0.45

        [Weights]
        overall_tech = 0.6
        overall_other = 0.4

        [Models]
        default_yolo_model = yolo12x.pt

        [JudgementLevels]
        excellent = 0.9
        good = 0.7
        fair = 0.5
        poor = 0.3
        ```

6.  **YOLO Model:**
    *   The script defaults to using the model specified in `config.ini` (e.g., `yolo12x.pt`).
    *   **Standard Models (e.g., `yolov8n.pt`):** If you specify a standard Ultralytics model name via `--model_path`, it will be downloaded automatically on the first run (internet connection required).
    *   **Custom/Default Model (e.g., `yolo12x.pt`):** If the model specified in `config.ini` (or your chosen model via `--model_path`) is *not* a standard auto-downloadable model name recognized by Ultralytics, you must:
        1.  Download or obtain the `.pt` model file.
        2.  Place it in the same directory as `photo_analyzer.py` OR provide its full path using the `--model_path` argument.
    *   **Class Names:** The script relies on class names being embedded within the YOLO model file (`.pt`) for human-readable object descriptions.

## Compatibility

*   **Python:** 3.7 or higher
*   **Python Packages:** See `requirements.txt`. Key dependencies are:
    *   `opencv-python`
    *   `numpy`
    *   `ultralytics`
    *   `tqdm`
*   **Operating System:** macOS, Linux, or Windows (with appropriate Python environment setup)

## Example Output

When run, the script provides a progress bar and logs information to the console. If `--verbose` is used, you'll get detailed JSON for each image:

```json
{
  "Sharpness": { "confidence": 0.85, "explanation": "Edges are sharp..." },
  "Focus Area": { "confidence": 0.90, "explanation": "Main subject ('person') is in sharp focus." },
  "Exposure": { "confidence": 0.92, "explanation": "Brightness is balanced..." },
  "Noise": { "confidence": 0.75, "explanation": "Minimal noise detected." },
  "Color Balance": { "confidence": 0.88, "explanation": "Colors are natural..." },
  "Dynamic Range": { "confidence": 0.95, "explanation": "Wide dynamic range..." },
  "description": "Image containing: person, dog, car.",
  "overall_confidence": 0.88,
  "judgement_description": "Overall technical quality is excellent. A main subject ('person')...",
  "judgement": "Excellent"
}
```

## Contributing

(Optional)

If you would like to contribute to this project, please feel free to submit pull requests or open issues on the GitHub repository.


## License

The Photo Quality Analyzer script and its associated documentation (excluding third-party dependencies) are released under the [MIT License](LICENSE). See the `LICENSE` file for more details.

**Third-Party Licenses:**
*   This project utilizes the `ultralytics` Python package, which is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
*   The YOLO models (e.g., `yolo12x.pt`, `yolov8n.pt`) used with this script are typically also licensed under AGPL-3.0 if obtained from Ultralytics.
Please ensure you comply with the terms of these licenses when using or distributing those components.

---
