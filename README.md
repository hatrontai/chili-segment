# Chili Pepper Phenotyping

Automated trait extraction from longitudinal chili-fruit sections using YOLOv7
object detection, YOLO11m-seg instance segmentation and geometric measurements.
Outputs cover fruit size, internal anatomy, seeds, shape and color.

## Setup

Use Python 3.11 and install the dependencies:

```bash
pip install -r requirements.txt
```

On Ubuntu/Debian, also install `libgl1`, `libglib2.0-0` and `libzbar0`.

## Model weights

Place the checkpoints in `weights/` (not included in Git):

| Model | File |
| --- | --- |
| Detection — Web and CLI | [yolov7_detection.onnx](https://drive.google.com/file/d/17DBxTxCuZt0Pi1XQAoFCthMxJjn7nhLp/view?usp=drive_link) |
| Segmentation — Web and CLI | [yolov11m_segment_final.pt](https://drive.google.com/file/d/1h57Z0VuCEqiLwBAD2MJME16eCRzTdjXz/view?usp=sharing) |

Download both model files above into `weights/` using the filenames shown.

Override paths with `CHILI_DETECTION_WEIGHT` and `CHILI_SEGMENT_WEIGHT`,
or CLI options `--weight_det` and `--weight_seg`.

## Run

Start the English HTML research demo at `http://localhost:7860` (FastAPI/Uvicorn; no frontend build needed):

```bash
python app.py
```

The single page introduces the research and provides the complete analysis workflow:

- Select a dataset sample or upload JPG/PNG/BMP/TIFF (up to 50 MB). An upload takes priority.
- Adjust the detector, threshold and wrinkle settings; expand **Advanced inference settings** for image size, crop and detection thresholds.
- Run analysis and inspect all six processing views, image zoom, color legends, per-fruit comparisons and the 34-column feature table.
- Download CSV, diagnostic manifest or a ZIP containing both. Unavailable measurements display as a dash and are returned as JSON `null`.

The top crop defaults to **540 px** for dataset headers. Set it to **0** for images without a header. Model loading and geometric measurements may take a few minutes on the first run. The page reports queued/running/completed/failed states, without estimated progress percentages; if polling loses connectivity, **Reconnect to analysis** checks the same task.

Inference runs one job at a time on a background worker. Uploads and artifacts are stored in a temporary directory per job, remain available for this server session, and are removed on normal shutdown. Closing or resetting the page does not cancel an already submitted job. Restarting the server clears all jobs; download results you want to keep. This is a local demo, with no accounts or persistent database.

Use `CHILI_SERVER_NAME` and `CHILI_SERVER_PORT` to override the bind address and port (defaults: `0.0.0.0:7860`). The older `GRADIO_SERVER_NAME`/`GRADIO_SERVER_PORT` variables remain fallbacks; `CHILI_*` takes priority.

### Local API

- `GET /api/samples`: image names/URLs, default settings and upload limit.
- `GET /api/samples/{name}`: one image from `data/`.
- `POST /api/jobs`: multipart form containing `sample_name`, optional image `upload`, and `options` as a JSON string. Returns HTTP 202 with `id`, `status` and `status_url`.
- `GET /api/jobs/{id}`: status, error or result (summary/configuration, images, captioned galleries, color legend, feature columns/rows and download URLs).
- `GET /api/jobs/{id}/files/{filename}`: generated images and downloads for a completed job. Input uploads and arbitrary server paths are not served.

`options` accepts `detector_backend`, `fruit_threshold_offset`, `wrinkle_tau_mm`, `seg_imgsz`, `crop_top`, `det_conf`, `det_iou` and `det_containment`. See `/docs` for endpoint documentation. Sample selection is restricted to `data/`; uploaded files must decode as images before inference is queued.

HTML/CSS/JavaScript live in `static/`; visualization helpers and export generation are in `src/web_service.py`. The web app and CLI share the same pipeline and measurement schema.

### Command line

Process an image or directory from the command line:

```bash
python predictor.py --input_path data/001_IT135673_1.jpg \
  --output-dir runs/predict --save_csv --save_img
```

Results include `output.csv`, `manifest.json` and segmentation overlays under
`images/`. The manifest records configuration, model hashes and diagnostics.
Use `python predictor.py --help` for all options. For images without the dataset
header, set `--det-crop-top 0` instead of the default 540-pixel crop.

## Tests

```bash
pip install httpx  # TestClient dependency for API tests
python -m unittest discover -s tests -p 'test_*.py' -v
```
