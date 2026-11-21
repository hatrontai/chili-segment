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

## Dataset

[Subset of Chili Dataset](https://drive.google.com/drive/folders/1RtOHoERPzH5632HS1NmBeejl5JY4zZw9?usp=sharing)
