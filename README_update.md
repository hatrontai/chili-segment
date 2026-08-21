# Chili Phenotyping V2 - Codebase Update

This update introduces major enhancements to the segmentation, detection and phenotypic feature extraction pipeline.

## 🚀 Key Updates & Features

### 1. Robust Phenotypic Trait Extraction (`src/object.py`)
- **Advanced Morphology:** Completely rewritten geometry extraction using OpenCV exact bitwise calculations for masks.
- **Stem/Placenta Separation (`find_armpit_points`):** Automatically extracts and isolates the stem from the placenta using contour curvature and angle analysis.
- **Complex Features Extracted:** 
  - `fruit_body_area`, `pericarp_area`, and `placenta_area`.
  - Continuous thickness monitoring (`thickness_avg`, `thickness_max`).
  - Skeletonization spine routing for tracking exact curvature (`curvature_index`) and straight-line asymmetric bends (`asymmetry_index`). 
  - Segmenting widths at 9 distinct locations across the chili spine (`width_9seg`).
- **Seed Optimization:** Enhanced counting to track strict seed geometry and `seed_density`.
- **Coloration Metrics:** Outputs chromatic redness (`redness_chroma`) and color homogeneity (`color_uniformity`).

### 2. Upgraded Adaptive Detection (`src/detect.py`)
- **HSV Space Thresholding (Otsu):** Moved from hardcoded Red-channel threshold to dynamic Otsu thresholding on the Saturation (HSV) channel, significantly improving segmentation of green and yellow peppers from dark backgrounds.
- **SAHI Integration (Slicing):** Integrated sliced overlapping inference (`AutoDetectionModel`) for handling massive scanned image resolutions perfectly, avoiding shrinking small peppers to oblivion.
- **Smart Aspect-Ratio Filters:** Built custom aspect-ratio and boundary constraints to confidently filter out cut-off color chips/labels at the edge without filtering out actual stubby pepper varieties.

### 3. Segmentation Enhancements (`src/segment.py` & `predictor.py`)
- **Dynamic Resolution Targeting:** Segmentation model now appropriately infers at custom sizes (`imgsz=960`), preserving checkpoint properties compared to the previously hardcoded `640px` shrinkage.
- **BGR Color Healing:** Passed direct RGB matrices natively through the correct `COLOR_RGB2BGR` pipeline beforehand to respect YOLO internal inversion behaviors.
- **Dynamic Confidence Mapping:** Lowered global inference threshold to `0.1` (to capture subtle layers) but forced a strict `0.35` threshold dedicated **only to Seeds** to resolve the existing 16.5% overcounting artifact.

## ⚙️ How to use
Run `predictor.py` as usual. The output CSV in `./runs/predict/output.csv` will automatically generate 20+ columns encompassing all the newly extracted physiological metrics.
