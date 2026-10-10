# Chili Pepper Phenotyping & Feature Extraction

This repository analyzes chili pepper images with a two-stage pipeline:

1. Detect each chili in the input image.
2. Segment each detected fruit and extract phenotypic features.

The current implementation uses:

- YOLOv8 segmentation for fruit component masks
- YOLOv7 ONNX detection on CPU by default, with a threshold-based fallback
- QR-code-based scale calibration when a QR code is present in the image

## Project Structure

- `app.py`: Gradio web UI
- `predictor.py`: CLI entry point for batch or single-image inference
- `src/detect.py`: chili detection logic
- `src/segment.py`: YOLOv8 segmentation and mask-to-polygon post-processing
- `src/object.py`: fruit/chili domain objects and feature calculations
- `src/geometric_features.py`: mesocarp skeleton, thickness ray-casting, and
  fruit-body contour wrinkle geometry
- `tests/test_geometric_features.py`: synthetic geometry and scale-fallback tests
- `data/`: sample images and matching annotations
- `weights/`: runtime ONNX detector and YOLOv8 segmentation weights
- `../yolov7_export/`: isolated official YOLOv7 export checkout and artifacts

## Requirements

The revised runtime was tested with Python 3.11 and these Python packages:

- `pandas`
- `pillow`
- `opencv-python`
- `numpy`
- `pyzbar`
- `shapely`
- `ultralytics`
- `onnxruntime`
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


## Pretrained Weights

The repository expects a segmentation weight file at:

```text
./weights/yolov8_segment_v2.pt
```

The default detector weight is `./weights/yolov7_detection.onnx`. Set
`CHILI_DETECTION_WEIGHT` to override it in the Web app, or
`CHILI_SEGMENT_WEIGHT` to override the segmentation path. A missing or invalid
ONNX model stops startup with an explicit error instead of silently switching
detectors.

The ONNX file is exported outside `code_base` from the official YOLOv7 source:

```bash
cd ../yolov7_export/yolov7
python export.py \
  --weights ../weights/yolov7_detection.pt \
  --img-size 640 640 \
  --batch-size 1 \
  --device cpu \
  --grid \
  --simplify
```

This produces a raw-output opset-12 model with input `[1,3,640,640]` and output
`[1,N,7]`. Validate it with `onnx.checker` before copying it to
`weights/yolov7_detection.onnx`. The export environment needs `onnx` and
`onnxsim`; application runtime needs only `onnxruntime`.

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
python predictor.py --input_path /path/to/image_or_directory
```

Useful flags:

- `--input_path`: single image path or directory of images
- `--detector-backend {yolov7_onnx,threshold}`: select detector; ONNX is default
- `--weight_det`: override the YOLOv7 ONNX path
- `--det-conf-thres`, `--det-iou-thres`: thresholds (defaults `0.70`, `0.30`)
- `--det-containment-thres`: nested-box IoM threshold (default `0.60`)
- `--det-input-size`: static ONNX side length (default `640`)
- `--det-crop-top`: top ROI offset (default `540` pixels)
- `--weight_seg`: segmentation weight path
- `--use_model_det` and `--name_model_det`: deprecated compatibility aliases
- `--mesocarp-wrinkle-segment-threshold-mm`: fruit-body contour chord threshold
  used for the wrinkle feature (default: `5.0` mm; independent of the red
  threshold)
- `--save_csv`: write results to `./runs/predict/output.csv`

If you use `--save_csv`, make sure `./runs/predict/` exists first, because the script does not create that directory automatically.

## Pipeline Details

### Detection

`src/detect.py` can run in two modes:

- `backend="yolov7_onnx"`: runs the raw graph using only `CPUExecutionProvider`
- `backend="threshold"`: uses a red-channel thresholding fallback

Both modes crop the top 540 pixels by default. ONNX preprocessing letterboxes
the remaining ROI to 640x640 with `(114,114,114)` padding, normalizes to
float32 NCHW, filters class `chili` (ID 1) using
`objectness * class_probability >= 0.70`, and applies NumPy NMS at IoU 0.30.
A lower-confidence nested box is also suppressed when intersection over the
smaller box area (IoM) reaches 0.60.

YOLOv7 candidates are trusted and are not rejected when segmentation classes
are absent. Threshold-detector candidates require at least one retained
segmentation mask from any class. When class `chili` is unavailable in YOLO
mask mode, the largest red-threshold contour becomes the measurement mask.
Component classes remain filtered at `0.10`, seed at `0.35`, and chili at
`0.01`; any retained subordinate mask is clipped to the measurement mask.
Bounding boxes are mapped back through the letterbox transform and the 540-pixel
offset. Input images therefore do not need to be 640x640. The fallback detector
is tuned for images matching the bundled dataset layout.

### Segmentation

`src/segment.py` wraps a YOLOv8 segmentation model and post-processes the masks
into polygons. Class ID `2` is the PCP class (Placenta + Pedicel), and
class `3` is `endocarp`. The class IDs remain unchanged from model training.

Every candidate must contain a nonempty selected measurement mask. ONNX detections
are trusted without requiring any learned class. Threshold-detector candidates
require at least one retained learned mask from any class. By default the
measurement mask is built by
red-channel thresholding, morphological closing, and retaining the largest
external contour. All subordinate masks are clipped to this main contour and
pixels outside it are rendered black, so a neighbouring fruit captured by the
50-pixel crop padding cannot affect measurements. Rejected candidates are not
included in `Fruits.items`, Web results, or CLI CSV output.

### Feature Extraction

`src/object.py` builds a `Fruits` object for each image and a `Chili` object for each detected fruit. The 17 primary traits use one canonical registry in `src/traits.py`:

| Trait | Symbol | Unit | Legacy source |
| --- | --- | --- | --- |
| Fruit Length | `FL` | mm | `length_fruit` |
| Mean Fruit Width | `FW_mean` | mm | `width_fruit` |
| Minimum Fruit Width | `FW_min` | mm | `width_fruit_stats["min"]` |
| Maximum Fruit Width | `FW_max` | mm | `width_fruit_stats["max"]` |
| Fruit Area | `FA` | mm² | `area_fruit` |
| Mesocarp Area | `MA` | mm² | `mesocarp_area` |
| Mean Mesocarp Thickness | `MT_mean` | mm | `mesocarp_thickness_stats["avg_mesocarp_thickness"]` |
| Minimum Mesocarp Thickness | `MT_min` | mm | `mesocarp_thickness_stats["min_mesocarp_thickness"]` |
| Maximum Mesocarp Thickness | `MT_max` | mm | `mesocarp_thickness_stats["max_mesocarp_thickness"]` |
| Placenta Area | `PA` | mm² | `placenta_area` |
| Total Seed Area | `SA` | mm² | `seed_stats["sum"]` |
| Seed Count | `SC` | seed | `number_seeds` |
| Mesocarp Area Fraction | `MAF` | dimensionless | `mesocarp_area_fraction` |
| Seed-to-Cavity Area Ratio | `SCR` | dimensionless | `seed_density` |
| Fruit Curvature Index | `FCI` | dimensionless | `curvature_index` |
| Fruit Asymmetry Index | `FAI` | dimensionless | `asymmetry_index` |
| Mesocarp Wrinkle Index | `MWI` | dimensionless | `mesocarp_wrinkle_index` |

Call `Chili.get_trait_values()` to obtain these values keyed by symbol. Web and CLI CSV outputs use the same human-readable columns, such as `Fruit Length (FL)` and `Seed-to-Cavity Area Ratio (SCR)`. Legacy attributes remain available for compatibility. Bounding boxes, 9-segment measurements, individual-seed statistics, color metrics, and scale metadata remain as extended output fields.

If a QR code is detected, its 10 mm side is used to estimate the
pixel-to-real-world scale. If QR decoding fails, runtime continues with the
documented dataset-mean estimate `0.094305839 mm/pixel`; Web and CSV output mark
the scale source as `dataset_mean_fallback`.

The PCP mask is separated into the entire green pedicel and the remaining
placenta.
Mesocarp thickness is sampled every 1 mm along the longest skeleton backbone.
Each sample uses the shortest valid straight line through the skeleton that
crosses one inner and one outer mesocarp edge. The mesocarp wrinkle index uses
the external contours of `mask_chili - (mask_pedicel | mask_outer)`.
Internal-hole contours are excluded; boundaries exposed by removing the pedicel
and outer are included. Closed contours retain full-chord cuts and a final
residual segment without tail rebalancing. A closed contour with no full-chord
intersection is partitioned into at least three equal-arc segments, each with
arc length no greater than the chord threshold. The default threshold remains
5 mm and MWI remains the mean of `arc/chord - 1`.

The public runtime names use `mask_pcp`, `mask_pedicel`, `mask_placenta`, and
`mask_mesocarp`. Feature outputs use `mesocarp_area`,
`mesocarp_thickness_stats` and `mesocarp_wrinkle_index`. Mesocarp Area Fraction
(`MAF = MA / FBA`) measures the fraction of the fruit-body cross-section occupied
by mesocarp. `FBA` excludes outer and pedicel; the mesocarp excludes endocarp,
outer, PCP, and seed. MAF uses unrounded pixel areas, is independent of spatial
scale, and is reported to four decimals in [0, 1]; an empty body yields `None`.
It is a 2D area fraction, not mass density or a validated predictor of yield.
MAF replaces FPR in the 17 primary Web/CLI traits. The old
`flesh_placenta_ratio` attribute and method remain available to existing callers
with their original meaning, but are absent from the canonical output.
The legacy `seed_density` attribute is the Seed-to-Cavity Area Ratio
(`SCR`), not a count-per-area density. Legacy wall/stem identifiers are not retained.

## Notes

- The repository includes notebooks for preprocessing and parameter exploration, but they are not required for runtime inference.
- The code is research-oriented and expects images in a format similar to the bundled samples.
- Some helper methods are present for visualization and analysis, but the main entry points are `app.py` and `predictor.py`.

### Regression sampling policy

Future image regressions should sample from `../jeju_segmentation/`, record a
random seed, and stratify for morphological diversity. At minimum, selection
should cover fruit count, relative polygon area, aspect ratio, circularity,
solidity, and extent instead of using only the small bundled `data/` subset.


## Runtime configuration, provenance and diagnostics (October 2026 revision)

Web and CLI use `src/pipeline.py` for configuration, feature rows and exports.
The default segmentation inference size is **640**, whole-fruit threshold offset
is **16**, and MWI chord threshold is **5 mm**. The fallback object detector uses
red-channel offset **22** and minimum contour area **10,000 px²**. These are
separate stages. Crop-top defaults to 540 px and crop padding is 50 px.

```bash
python predictor.py --input_path data/001_IT135673_1.jpg \
  --seg-imgsz 640 --det-crop-top 540 --output-dir runs/review \
  --save_csv --save_img
```

`--output-dir` is created automatically. `--save_csv` writes `output.csv` without
an extra pandas index; `--save_img` writes oriented segmentation overlays under
`images/<input-stem>/`. Each run writes `manifest.json`, even without CSV. Use a
separate output directory for each run because names within a directory are
replaced. The Web UI downloads a ZIP containing the same numeric table and a
manifest, and exposes inference size, crop-top and detector thresholds.

The manifest records effective settings, input/checkpoint SHA-256 hashes, source
hashes, Git revision/dirty state, direct dependency versions, image runtime,
scale source, candidate rejection reasons and per-fruit diagnostics. Old CSV
trait columns/values are preserved; diagnostic statuses are separate. A
`computed` status means that an algorithm returned a value, not that a physical
reference validated that value. Failure-related legacy zeros retain their
numeric value and are identified by status/reasons. Empty seed masks do not
prove biological seed absence. Thickness diagnostics expose valid/filtered/used
sample counts and the prefilter fallback. MWI exposes contour/segment counts.
Threshold-mask confidence 1.0 is a synthetic presence flag, not a probability.

The pipeline measures prepared camera images of longitudinal sections. FL/FW
exclude outer but may include pedicel; FA includes the complete selected mask.
Canonical FW uses all nonempty rows after a 5% spatial trim, rather than the
auxiliary 9-segment output. FCI uses the rounded discrete FL and can be below
one for a straight finite mask. MWI uses closed external fruit-body contours;
raster orientation, global curvature and cut edges can contribute. Color
Uniformity is population SD of the OpenCV zero-centered a channel: higher means
less uniform. Its region includes endocarp pixels and has no white-balance
normalization. Names mesocarp/endocarp are operational annotation labels.

`requirements-tested.txt` records direct packages in the tested environment;
the torch CUDA build string is provenance for that environment, not a portable
CPU wheel requirement. Use `requirements.txt` for a platform-appropriate install
and retain its resolved versions in manifests. Linux QR decoding needs libzbar;
otherwise the documented dataset-mean scale fallback is used.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/evaluate_measurement_masks.py --output ../review_artifacts/mask_diagnostics.json
```

Mask diagnostics compare selected threshold masks and optional cleaned YOLO
masks on the bundled sample image subset using current five-class annotations
from `../jeju_segmentation/`. Historical bundled JSON files are not whole-fruit
references. Mismatched annotation dimensions are skipped without guessing a
transform. One-to-one matching uses IoU >= 0.5; Dice and boundary F (2 original
pixel tolerance) are reported for matched objects, with FP/FN counts separately.
Split membership and color strata are unknown. This diagnostic is not a held-out
paper experiment and does not replace reported results or physical trait validation.
