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
- `src/geometric_features.py`: mesocarp skeleton, thickness ray-casting, and
  outer-mesocarp wrinkle geometry
- `tests/test_geometric_features.py`: synthetic geometry and scale-fallback tests
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

The Gradio pipeline inspector also lets you select one of the bundled images in
`data/`. Its tabs expose the intermediate detection ROI, threshold mask,
bounding boxes, raw and orientation-normalized fruit crops, segmentation
overlays, individual class masks, and the final feature table. When both a
sample and an uploaded image are provided, the uploaded image takes priority.

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
- `--mesocarp-wrinkle-segment-threshold-mm`: outer-mesocarp chord threshold
  used for the wrinkle feature (default: `5.0` mm; independent of the red
  threshold)
- `--save_csv`: write results to `./runs/predict/output.csv`

If you use `--save_csv`, make sure `./runs/predict/` exists first, because the script does not create that directory automatically.

## Pipeline Details

### Detection

`src/detect.py` can run in two modes:

- `model_name="yolo"`: uses a YOLO detection model
- `model_name="none"`: uses a red-channel thresholding fallback

The fallback detector is tuned for images that match the dataset layout used in this repo. It crops the top 540 pixels before detection.

### Segmentation

`src/segment.py` wraps a YOLOv8 segmentation model and post-processes the masks
into polygons. Class ID `2` is the PCP class (Placenta + Pedicel), and
class `3` is `endocarp`. The class IDs remain unchanged from model training.

Every detector crop must also contain a non-empty YOLO class `chili` mask
(class ID `4`) before it is exposed as a fruit. In threshold mode, that YOLO
mask is used only to validate the candidate; the measurement mask is built by
red-channel thresholding, morphological closing, and retaining the largest
external contour. All subordinate masks are clipped to this main contour and
pixels outside it are rendered black, so a neighbouring fruit captured by the
50-pixel crop padding cannot affect measurements. Rejected candidates are not
included in `Fruits.items`, Web results, or CLI CSV output.

### Feature Extraction

`src/object.py` builds a `Fruits` object for each image and a `Chili` object for each detected fruit. It extracts:

- bounding box width and height
- fruit width
- fruit length
- area
- redness
- mesocarp area and flesh-to-placenta ratio
- mesocarp thickness (minimum, average, and maximum)
- mesocarp wrinkle index (mean `arc/chord - 1`)

If a QR code is detected, its 10 mm side is used to estimate the
pixel-to-real-world scale. If QR decoding fails, runtime continues with the
documented dataset-mean estimate `0.094305839 mm/pixel`; Web and CSV output mark
the scale source as `dataset_mean_fallback`.

The PCP mask is separated into the entire green pedicel and the remaining
placenta.
Mesocarp thickness is sampled every 1 mm along the longest skeleton backbone.
Each sample uses the shortest valid straight line through the skeleton that
crosses one inner and one outer mesocarp edge. The mesocarp wrinkle index uses
only the true outer mesocarp edge and excludes the pedicel region.

The public runtime names use `mask_pcp`, `mask_pedicel`, `mask_placenta`, and
`mask_mesocarp`. Feature outputs use `mesocarp_area`,
`mesocarp_thickness_stats` and `mesocarp_wrinkle_index`. The separate
`flesh_placenta_ratio` feature retains its original definition
(`A_flesh / A_placenta`): its flesh mask excludes outer, PCP, and seed while
retaining endocarp. It is reported only when the cleaned placenta mask contains
at least 10 pixels. Legacy wall/stem identifiers are not retained.

## Notes

- The repository includes notebooks for preprocessing and parameter exploration, but they are not required for runtime inference.
- The code is research-oriented and expects images in a format similar to the bundled samples.
- Some helper methods are present for visualization and analysis, but the main entry points are `app.py` and `predictor.py`.
