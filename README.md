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
- `src/pipeline.py`: shared runtime configuration, provenance and exports
- `src/traits.py`: canonical trait registry for Web and CLI output
- `tests/`: regression tests and measurement/detection evaluation scripts
- `data/`: sample images and matching annotations
- `weights/`: runtime ONNX detector and YOLOv8 segmentation weights

This repository contains runtime code, regression tests and bundled sample data.
Historical notebooks, progress reports and the paper snapshot were archived
outside the repository during the November 2026 cleanup. They remain available
in earlier Git commits; they are not runtime dependencies. Model weights stay
local under `weights/` and are excluded from Git.

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

The visualization helpers also require `matplotlib`; it is included in
`requirements.txt`.

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

The default segmentation checkpoint depends on the entry point:

| Entry point | Local segmentation checkpoint |
| --- | --- |
| Web (`app.py`) | `weights/yolov8_segment_v3.pt` |
| CLI (`predictor.py`) and `PipelineConfig` | `weights/yolov8_segment_v2.pt` |

The default detector weight is `./weights/yolov7_detection.onnx`. Set
`CHILI_DETECTION_WEIGHT` to override it in the Web app, or
`CHILI_SEGMENT_WEIGHT` to override the segmentation path. A missing or invalid
ONNX model stops startup with an explicit error instead of silently switching
detectors.

The optional ONNX export step uses a separate checkout of the official YOLOv7
source. Run this in that checkout with the checkpoint's absolute path:

```bash
cd /path/to/yolov7
python export.py \
  --weights /path/to/yolov7_detection.pt \
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
anatomical segmentation mask. The whole-fruit measurement mask is always
created independently from the red channel by adaptive thresholding, closing,
and retaining the largest external contour. Learned class ID `4` from legacy
five-class weights is ignored; only seed, outer, PCP, and endocarp masks are
used from the segmentation model. Component classes are filtered at `0.10`
and seed at `0.35`; retained anatomical masks are clipped to the threshold
whole-fruit mask.
Bounding boxes are mapped back through the letterbox transform and the 540-pixel
offset. Input images therefore do not need to be 640x640. The fallback detector
is tuned for images matching the bundled dataset layout.

### Segmentation

`src/segment.py` wraps a YOLOv8 segmentation model and post-processes the masks
into polygons for four anatomical classes: seed (`0`), outer (`1`), PCP
(Placenta + Pedicel, `2`), and endocarp (`3`). The whole-fruit measurement
mask is stored separately as `fruit_mask` and is not a YOLO segmentation
class. Existing five-class weights remain loadable; their class ID `4`
predictions are discarded.

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

`src/object.py` builds a `Fruits` object for each image and a `Chili` object for each detected fruit. The 19 primary traits use one canonical registry in `src/traits.py`:

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
| Maximum Leftward Deviation | `D_L` | dimensionless | `deviation_stats["D_L"]` |
| Maximum Rightward Deviation | `D_R` | dimensionless | `deviation_stats["D_R"]` |
| Mean Signed Deviation | `D_mean` | dimensionless | `deviation_stats["D_mean"]` |
| Mesocarp Wrinkle Index | `MWI` | dimensionless | `mesocarp_wrinkle_index` |

Call `Chili.get_trait_values()` to obtain these values keyed by symbol. Web and CLI CSV outputs use the same human-readable columns, such as `Fruit Length (FL)` and `Seed-to-Cavity Area Ratio (SCR)`. Legacy attributes remain available for compatibility. Bounding boxes, 9-segment measurements, individual-seed statistics, color metrics, and scale metadata remain as extended output fields.

Total Seed Area (`SA`) is the area of the union of retained YOLO seed masks,
scaled by `ratio**2` to mm² and rounded to two decimals. Overlapping pixels
contribute once. Seed Count (`SC`) counts the nonempty YOLO seed instances after
confidence filtering and clipping to the fruit mask. No automatic watershed
fallback changes this count. Individual seed min/max/avg/std still use each
instance's area, so `SA` need not equal `SC * avg`. The previous watershed
measurement remains available explicitly via `get_seed_stats(method="watershed")`
for comparisons.

For `SCR`, the cavity is the current endocarp mask:
`SCR = |M_seed| / (|M_endocarp| + epsilon_SCR)`, with `epsilon_SCR = 1e-6`
in pixel-count units. This prevents division by zero without a separate branch.
An empty seed mask returns zero; an empty endocarp with detected seed pixels
produces a large finite ratio. The ratio is not clamped.

The historical paper snapshot was archived outside this runtime repository;
it can also be recovered from earlier Git commits.

Fruit width follows the longitudinal-slice protocol used by this project's
pepper phenotyping method: divide the segmented fruit mask height into nine
equal sections using ten horizontal scanlines, measure the eight interior
chords from the leftmost to rightmost fruit-mask pixel, and report their mean
as width. `FW_min` and `FW_max` are the minimum and maximum of those same eight
chords. The spine-derived curve remains used for fruit length and curvature;
it is not used to invent a local width direction for broad or lobed pepper
silhouettes.

If a QR code is detected, its 10 mm side is used to estimate the
pixel-to-real-world scale. If QR decoding fails, runtime continues with the
documented dataset-mean estimate `0.094305839 mm/pixel`; Web and CSV output mark
the scale source as `dataset_mean_fallback`.

The PCP mask is separated into the entire green pedicel and the remaining
placenta.
Mesocarp thickness is sampled every 1 mm along the longest skeleton backbone.
Each sample uses the shortest valid straight line through the skeleton that
crosses one inner and one outer mesocarp edge. The mesocarp wrinkle index uses
the external contours of `mask_fruit - (mask_pedicel | mask_outer)`.
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
MAF replaces FPR in the primary Web/CLI traits. The old
`flesh_placenta_ratio` attribute and method remain available to existing callers
with their original meaning, but are absent from the canonical output.
The legacy `seed_density` attribute is the Seed-to-Cavity Area Ratio
(`SCR`), not a count-per-area density. Legacy wall/stem identifiers are not retained.

## Notes

- Historical preprocessing and parameter-exploration notebooks were archived outside the runtime repository.
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
prove biological seed absence. Thickness diagnostics expose sampled, successful and used sample counts. MWI exposes contour/segment counts.
Threshold-mask confidence 1.0 is a synthetic presence flag, not a probability.

The pipeline measures prepared camera images of longitudinal sections. FL uses the fruit-body mask (whole fruit minus outer and separated pedicel);
FCI uses the same body spine and FL. Failed pedicel separation can leave pedicel
pixels in the body. Signed deviation uses this same body mask and spine. FA measures the body mask after excluding outer and pedicel. Canonical FW uses eight interior
horizontal chords of the whole-fruit mask, as described above. FCI uses the rounded discrete FL and can be slightly above
zero for a straight finite mask. MWI uses closed external fruit-body contours;
raster orientation, global curvature and cut edges can contribute. Color
Uniformity is population SD of the OpenCV zero-centered a channel: higher means
less uniform. Its region includes endocarp pixels and has no white-balance
normalization. Names mesocarp/endocarp are operational annotation labels.

Run manifests record installed direct dependency versions. A torch CUDA build
string describes its environment and is not a portable CPU wheel requirement.
Use `requirements.txt` for a platform-appropriate install
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


### Signed body deviation (replaces FAI)

`get_deviation_stats()` reuses `self.spine` from FL/FCI, fitted on whole fruit
minus outer and pedicel. The straight chord joining its endpoints defines the
reference axis. At each nonempty spine row, clamp that axis to the row's left
and right boundaries, then measure W_L and W_R in pixels.
The signed deviation is d = (W_R - W_L)/(W_R + W_L + epsilon_A), with
epsilon_A = 1e-6 pixel. D_L = min(0, min(d)), D_R = max(0, max(d)), and D_mean
is the arithmetic mean across all nonempty spine rows. Single-pixel rows
contribute zero and count in that mean; empty rows do not. Negative means left,
positive means right. Opposite deviations can cancel in the mean. No usable
spine yields zeros, marked unavailable by measurement diagnostics.
Values are rounded to four decimal places, so output may display -1 or 1 even
though epsilon makes the unrounded values strictly between those limits.

Web/CLI outputs now contain 19 canonical traits: D_L, D_R and D_mean replace
the old FAI column. Consumers of old CSV schemas must update their mappings;
previous FAI data cannot be relabeled as signed deviation. Re-run measurements
to generate the new results. FCI uses 1 - (D_chord + 1e-6)/(FL + 1e-6), with
lengths in mm; smaller values mean straighter fruit bodies.


### Smoothed-mask thickness — 2026-10-26

Thickness boundaries, signed-distance field and skeleton all use the same
morphologically smoothed mesocarp mask. No raw-mask skeleton fallback is used.
All successfully measured samples enter the min/mean/max statistics; spatial
and local-outlier screening and the prefilter fallback are no longer applied.
The two-stage angular search remains the numerical implementation of shortest
valid boundary-to-boundary section selection. Mesocarp area continues to use
the original cleaned mask. Recompute thickness results after this change.

### Fruit area FA — 2026-10-26

FA now equals gamma² times the foreground pixel count in the fruit-body mask
(whole fruit minus outer and pedicel). The legacy extended `fruit_body_area`
field reports the same area. MAF continues to divide mesocarp pixels by body
pixels before rounding; it now corresponds to MA/FA. Existing FA results must
be recomputed before using them with the revised definition.

### Bounded MWI — 2026-10-26

MWI is the arithmetic mean of 1 - L_chord/L_arc over valid body-contour
segments. The 5 mm chord threshold and contour segmentation remain in use.
Unrounded scores are in [0,1); reported values are rounded to five decimals.
The segment field is `wrinkle_score`, and the diagnostic maximum is
`maximum_segment_wrinkle`. Recompute old MWI results for this definition.
