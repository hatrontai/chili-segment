# Chili Pepper Phenotyping & Feature Extraction

This repository analyzes chili pepper images with a two-stage pipeline:

1. Detect each chili in the input image.
2. Segment each detected fruit and extract phenotypic features.

The current implementation uses:

- YOLOv8 segmentation for fruit component masks
- A YOLO-based detector when enabled, or a threshold-based fallback detector
- QR-code-based scale calibration when a QR code is present in the image

## Project Structure

- `app.py`: Gradio web UI
- `predictor.py`: CLI entry point for batch or single-image inference
- `src/detect.py`: chili detection logic
- `src/segment.py`: YOLOv8 segmentation and mask-to-polygon post-processing
- `src/object.py`: fruit/chili domain objects and feature calculations
- `data/`: sample images and matching annotations
- `pretrain/`: expected location for pretrained weights
- `Dockerfile`: container setup for the web app

## Requirements

The project expects Python 3.10 and these Python packages:

- `pandas`
- `pillow`
- `opencv-python`
- `numpy`
- `pyzbar`
- `shapely`
- `ultralytics`
- `gradio`
- `scipy`

The code also relies on a plotting library for visualization helpers, so `matplotlib` should be installed as well if you use the notebook or display functions.

System packages commonly needed on Linux:

- `libgl1`
- `libglib2.0-0`
- `libzbar0`

## Setup

### Local

```bash
pip install -r requirements.txt
```

If you are running on Ubuntu or Debian, you may also need:

```bash
sudo apt-get install libgl1 libglib2.0-0 libzbar0
```

### Docker

```bash
docker build -t chili-phenotyping-app .
docker run -p 7860:7860 chili-phenotyping-app
```

The container starts the Gradio app on port `7860`.

## Pretrained Weights

The repository expects a segmentation weight file at:

```text
./pretrain/yolov8_segment_v1.pt
```

If you want to use YOLO-based detection, provide a detection weight file and pass `--use_model_det` with `--weight_det` to the CLI.

## Web App

Run the Gradio interface with:

```bash
python app.py
```

Upload a chili image and the app returns a table of extracted features for each detected fruit.

## CLI Pipeline

Run inference from the command line with:

```bash
python predictor.py --input_path /path/to/image_or_directory --weight_seg ./pretrain/yolov8_segment_v1.pt
```

Useful flags:

- `--input_path`: single image path or directory of images
- `--use_model_det`: use the YOLO detector instead of threshold detection
- `--weight_det`: detection weight path when `--use_model_det` is enabled
- `--weight_seg`: segmentation weight path
- `--save_csv`: write results to `./runs/predict/output.csv`

If you use `--save_csv`, make sure `./runs/predict/` exists first, because the script does not create that directory automatically.

## Evaluation / Debug Output

Run the pipeline on a folder and save intermediate output for every detected chili:

```bash
python evaluate.py \
  --input_path ./data \
  --output_dir ./runs/evaluation \
  --weight_seg ./pretrain/yolov8_segment_v1.pt \
  --pixel_per_cm 120
```

Use either `--pixel_per_cm`, or both `--reference_length_px` and
`--reference_length_cm`, to export centimeter measurements. Without a scale,
pixel measurements are still exported.

```text
runs/evaluation/
├── features.csv
├── errors.csv
└── <image_name>/
    ├── step1_detection_bboxes.jpg
    └── chili_000/
        ├── step1_debug/
        ├── step2_masks/
        ├── step3_length/
        ├── step4_color/
        ├── step5_measurement/
        └── debug_summary.jpg
```

The mask step saves every class before and after cleanup. The length step
compares the longest pruned skeleton path, mask bounding-box length, and
contour perimeter. The color step exports HSV/LAB statistics and a red-pixel
ratio after removing strong shadows.

Useful tuning options:

- `--min_component_area`: minimum connected-component area retained in masks
- `--min_branch_length`: skeleton branches shorter than this are pruned
- `--recursive`: include images in nested input folders
- `--fail_fast`: stop instead of recording an image error and continuing

## Pipeline Details

### Detection

`src/detect.py` can run in two modes:

- `model_name="yolo"`: uses a YOLO detection model
- `model_name="none"`: uses a red-channel thresholding fallback

The fallback detector is tuned for images that match the dataset layout used in this repo. It crops the top 540 pixels before detection.

### Segmentation

`src/segment.py` wraps a YOLOv8 segmentation model and post-processes the masks into polygons. The code focuses on class `3` (`endocarp`) for mask cleanup and polygon reconstruction.

### Feature Extraction

`src/object.py` builds a `Fruits` object for each image and a `Chili` object for each detected fruit. It extracts:

- bounding box width and height
- fruit width
- fruit length
- area
- redness
- wrinkle score

If a QR code is detected, its size is used to estimate the pixel-to-real-world scale.

## Notes

- The repository includes notebooks for preprocessing and parameter exploration, but they are not required for runtime inference.
- The code is research-oriented and expects images in a format similar to the bundled samples.
- Some helper methods are present for visualization and analysis, but the main entry points are `app.py` and `predictor.py`.
