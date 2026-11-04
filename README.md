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
- Follow the live execution graph while stages finish and publish their outputs. Click a node or select a candidate to inspect it; **Follow live** returns to the active step.
- Inspect all six processing views, image zoom, color legends, per-fruit comparisons and the 34-column feature table.
- Download CSV, diagnostic manifest or a ZIP containing both. Unavailable measurements display as a dash and are returned as JSON `null`.

The top crop defaults to **540 px** for dataset headers. Set it to **0** for images without a header. Model loading and geometric measurements may take a few minutes on the first run. The page reports queued/running/completed/failed states and streams real stage events over SSE, without estimated progress percentages; if polling loses connectivity, **Reconnect to analysis** checks the same task.

Inference runs one job at a time on a background worker. Uploads and artifacts are stored in a temporary directory per job, remain available for this server session, and are removed on normal shutdown. Closing or resetting the page does not cancel an already submitted job. Restarting the server clears all jobs; download results you want to keep. This is a local demo, with no accounts or persistent database.

Use `CHILI_SERVER_NAME` and `CHILI_SERVER_PORT` to override the bind address and port (defaults: `0.0.0.0:7860`). The older `GRADIO_SERVER_NAME`/`GRADIO_SERVER_PORT` variables remain fallbacks; `CHILI_*` takes priority.

### Looping hero film

The opening image frame now contains a locally rendered HyperFrames demo (`static/media/chili-demo.mp4`): 14 seconds, 1440×1080, 30 fps. It demonstrates the real specimen, detection, anatomy masks and measurements with camera movement, a scan, staggered boxes and moving result panels. It autoplays muted, loops continuously and uses inline playback on mobile. A Pause/Play button is available; reduced-motion preferences start the video paused, and a matching poster is shown if video playback fails.

The editable source and rendering instructions are in `../chili-demo-video/`. Replacing the MP4 does not require rebuilding the frontend or loading inference models.

### Live graph and Replay

The graph follows **Input & ROI → Model Setup → Scale Calibration → Detection → Padded Crop → Segmentation & Orientation → Anatomy Masks → Trait Extraction → Results & Comparison → Export**. Dashed edges show data dependencies; the dotted return edge repeats the per-candidate steps. Orientation includes the initial segmentation prediction and a second prediction only when rotation is needed, exactly as in the original pipeline.

Running nodes illuminate; completed nodes turn green, failed nodes turn red and unneeded steps are marked skipped. The output panel includes stage images, calibration/model details, measurements or downloads as soon as each becomes available. Candidate numbering is independent from accepted fruit IDs: a rejected candidate is recorded without consuming a Chili ID. If no candidates are found, the per-fruit stages are skipped and an empty table/CSV is exported.

After success or failure, use **Replay**, **Pause**, **Restart**, the timeline slider and **0.5× / 1× / 2×** speed. Replay uses recorded events and artifacts only; it never submits another analysis. For presentation, intervals are condensed/clamped to 0.6–2 seconds before applying playback speed. The displayed elapsed times remain the original recorded times. Scrubbing hides outputs that were not available at the selected event; the final results return at the end or with **Show recorded run**.

Completed stage artifacts stay available even if a later step fails. SSE resumes from the last event ID after a dropped connection. Trace history and artifacts last for the current server session, with the same cleanup behavior as other results. Motion is disabled when the browser requests reduced motion.

### Local API

- `GET /api/samples`: image names/URLs, default settings and upload limit.
- `GET /api/samples/{name}`: one image from `data/`.
- `POST /api/jobs`: multipart form containing `sample_name`, optional image `upload`, and `options` as a JSON string. Returns HTTP 202 with `id`, `status`, `status_url`, `events_url` and `trace_url`.
- `GET /api/jobs/{id}`: status, error or result (summary/configuration, images, captioned galleries, color legend, feature columns/rows and download URLs), plus a `progress` snapshot containing the latest node states, candidate outcomes and sequence number.
- `GET /api/jobs/{id}/events`: SSE stream (`event: progress`, `id: <seq>`); resume with `Last-Event-ID` or `?after=<seq>`. Events carry `seq`, `elapsed_seconds`, `node`/`state`, optional candidate context, messages and ready `outputs`. Candidate/job lifecycle events use `kind` instead of a node.
- `GET /api/jobs/{id}/trace`: recorded events in sequence order for inspection and Replay, including partial/failed runs.
- `GET /api/jobs/{id}/files/{filename}`: published images and downloads, including outputs from running or failed jobs. Files must be finished and explicitly registered before they are served. Input uploads, partial/unpublished files and arbitrary server paths are not served.

`options` accepts `detector_backend`, `fruit_threshold_offset`, `wrinkle_tau_mm`, `seg_imgsz`, `crop_top`, `det_conf`, `det_iou` and `det_containment`. See `/docs` for endpoint documentation. Sample selection is restricted to `data/`; uploaded files must decode as images before inference is queued.

HTML/CSS/JavaScript live in `static/`; visualization helpers and export generation are in `src/web_service.py`. The optional `on_progress(event)` callback on `run_image`, `Fruits` and `Chili` observes real execution boundaries. The journal in `src/progress.py` stores immutable, sequenced events and published artifact references; arrays/models remain internal to the worker. The web app and CLI share the same pipeline and measurement schema.

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
