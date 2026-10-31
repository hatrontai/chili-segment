import os
import math
import time
import argparse
import torch
import pandas as pd
from pathlib import Path

from src.detect import (
    DEFAULT_CONFIDENCE,
    DEFAULT_CONTAINMENT,
    DEFAULT_CROP_TOP,
    DEFAULT_INPUT_SIZE,
    DEFAULT_IOU,
    DetectModel,
)
from src.pipeline import (PipelineConfig, build_models, run_image, feature_row,
                          feature_columns, create_manifest, image_diagnostics,
                          write_exports, save_overlays)


BASE_DIR = Path(__file__).resolve().parent
WEIGHT_MODEL_DETECT = os.getenv(
    "CHILI_DETECTION_WEIGHT",
    str(BASE_DIR / "weights" / "yolov7_detection.onnx"),
)
WEIGHT_MODEL_SEGMENT = os.getenv(
    "CHILI_SEGMENT_WEIGHT", str(BASE_DIR / "weights" / "yolov11m_segment_final.pt")
)
DEVICE = 0 if torch.cuda.is_available() else "cpu"

PREDICTOR_COLUMNS = feature_columns(include_root=True)

def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Chili Pepper Segmentation")

    parser.add_argument("--input_path", type=str, required=True, help="Path or dir to input image")
    parser.add_argument(
        "--detector-backend",
        choices=["yolov7_onnx", "threshold"],
        default="yolov7_onnx",
        help="Object detector backend (default: YOLOv7 ONNX on CPU)",
    )
    parser.add_argument(
        "--use_model_det",
        action="store_true",
        help="Deprecated alias that forces --detector-backend yolov7_onnx",
    )
    parser.add_argument(
        "--name_model_det",
        choices=["yolo", "yolov7_onnx", "none", "threshold"],
        default=None,
        help="Deprecated detector alias: yolo=ONNX, none=threshold",
    )
    parser.add_argument("--name_model_seg", default='yolo', type=str, help="Name model segmentation")
    parser.add_argument("--device", default=DEVICE, help="set device to run model")
    parser.add_argument("--weight_det", default=WEIGHT_MODEL_DETECT, type=str, required=False, help="Weight pretrain of model object detection for get bbox chili")
    parser.add_argument("--weight_seg", default=WEIGHT_MODEL_SEGMENT, type=str, required=False, help="Weight pretrain of model segmentation")
    parser.add_argument(
        "--det-conf-thres", default=DEFAULT_CONFIDENCE, type=float
    )
    parser.add_argument("--det-iou-thres", default=DEFAULT_IOU, type=float)
    parser.add_argument(
        "--det-containment-thres",
        default=DEFAULT_CONTAINMENT,
        type=float,
        help="Suppress a lower-confidence box when IoM reaches this value",
    )
    parser.add_argument(
        "--det-input-size", default=DEFAULT_INPUT_SIZE, type=int
    )
    parser.add_argument("--det-crop-top", default=DEFAULT_CROP_TOP, type=int)
    parser.add_argument(
        "--fruit-threshold-offset",
        default=16,
        type=int,
        help="Offset added to the dominant red-channel value in threshold mode",
    )
    parser.add_argument(
        "--mesocarp-wrinkle-segment-threshold-mm",
        default=5.0,
        type=positive_float,
        help="Fruit-body contour chord threshold in mm used by the wrinkle feature",
    )
    parser.add_argument("--seg-imgsz", default=640, type=int, help="Segmentation inference size")
    parser.add_argument("--output-dir", default="./runs/predict", type=Path, help="CSV, manifest and overlay directory")
    parser.add_argument("--save_csv", action="store_true", help="save output result")
    parser.add_argument("--save_img", action="store_true", help="save image output")

    return parser.parse_args(argv)

def config_from_args(args):
    backend = args.detector_backend
    if args.name_model_det is not None:
        backend = "threshold" if args.name_model_det in {"none", "threshold"} else "yolov7_onnx"
    if args.use_model_det:
        backend = "yolov7_onnx"
    if args.name_model_seg != "yolo":
        raise ValueError("Only yolo segmentation is supported")
    return PipelineConfig(
        detector_backend=backend, weight_det=args.weight_det, weight_seg=args.weight_seg,
        device=str(args.device), det_conf=args.det_conf_thres,
        det_iou=args.det_iou_thres, det_containment=args.det_containment_thres,
        det_input_size=args.det_input_size, det_crop_top=args.det_crop_top,
        seg_imgsz=args.seg_imgsz,
        fruit_threshold_offset=args.fruit_threshold_offset,
        wrinkle_tau_mm=args.mesocarp_wrinkle_segment_threshold_mm,
    )


def inference(args):
    config = config_from_args(args)
    source = Path(args.input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in extensions) if source.is_dir() else [source]
    if not paths:
        raise ValueError(f"No supported images in {source}")
    manifest = create_manifest(config, paths)
    models = build_models(config)
    rows = []
    for path in paths:
        started = time.perf_counter()
        try:
            fruits = run_image(path, config, models)
            rows.extend(feature_row(chili, path.stem) for chili in fruits.items)
            entry = image_diagnostics(fruits)
            entry["elapsed_seconds"] = time.perf_counter() - started
            manifest["images"].append(entry)
            if args.save_img:
                save_overlays(fruits, args.output_dir)
        except (ValueError, OSError) as exc:
            manifest["errors"].append({"input": str(path.resolve()), "message": str(exc)})
            print(f"Failed {path.name}: {exc}")
    results = pd.DataFrame(rows, columns=PREDICTOR_COLUMNS)
    write_exports(args.output_dir, results, manifest, save_csv=args.save_csv)
    results.attrs["manifest"] = manifest
    return results

def main():
    args = parse_args()

    print("Arguments:", args)
    results = inference(args)
    print(results)
    if results.attrs["manifest"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
