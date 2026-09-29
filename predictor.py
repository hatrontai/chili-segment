import os
import matplotlib.pyplot as plt
import shutil
import glob
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
from src.segment import SegmentModel
from src.object import Fruits, Chili


BASE_DIR = Path(__file__).resolve().parent
WEIGHT_MODEL_DETECT = os.getenv(
    "CHILI_DETECTION_WEIGHT",
    str(BASE_DIR / "weights" / "yolov7_detection.onnx"),
)
WEIGHT_MODEL_SEGMENT = os.getenv(
    "CHILI_SEGMENT_WEIGHT", str(BASE_DIR / "weights" / "yolov8_segment_v2.pt")
)
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def parse_args():
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
        "--chili_method",
        choices=["threshold", "yolo"],
        default="threshold",
        help="Method used to build class chili; other classes still use YOLO",
    )
    parser.add_argument(
        "--chili_threshold",
        default=16,
        type=int,
        help="Offset added to the dominant red-channel value in threshold mode",
    )
    parser.add_argument(
        "--mesocarp-wrinkle-segment-threshold-mm",
        default=5.0,
        type=positive_float,
        help="Outer-mesocarp chord threshold in mm used by the wrinkle feature",
    )
    parser.add_argument("--save_csv", action="store_true", help="save output result")
    parser.add_argument("--save_img", action="store_true", help="save image output")

    return parser.parse_args()

def inference(args):
    detector_backend = args.detector_backend
    if args.name_model_det is not None:
        detector_backend = (
            "threshold"
            if args.name_model_det in {"none", "threshold"}
            else "yolov7_onnx"
        )
    if args.use_model_det:
        detector_backend = "yolov7_onnx"

    model_detect = DetectModel(
        backend=detector_backend,
        weight_path=args.weight_det,
        conf_threshold=args.det_conf_thres,
        iou_threshold=args.det_iou_thres,
        containment_threshold=args.det_containment_thres,
        input_size=args.det_input_size,
        crop_top=args.det_crop_top,
    )

    model_segment = SegmentModel(
        weight_path=args.weight_seg,
        model_name=args.name_model_seg,
        device=args.device,
        chili_method=args.chili_method,
        chili_threshold=args.chili_threshold,
    )
    
    data_dir = args.input_path

    if not os.path.exists(data_dir):
        print("input path is not exist", data_dir)

    if os.path.isdir(data_dir):
        image_paths = glob.glob(f"{data_dir}/*.png") + \
              glob.glob(f"{data_dir}/*.jpg") + \
              glob.glob(f"{data_dir}/*.jpeg") + \
              glob.glob(f"{data_dir}/*.PNG") + \
              glob.glob(f"{data_dir}/*.JPG") + \
              glob.glob(f"{data_dir}/*.JPEG")
    else:
        image_paths = [data_dir]
    
    rows = []
    for image_path in image_paths:
        name_image = Path(image_path).stem
        chilis = Fruits(
            image_path,
            model_detect=model_detect,
            model_segment=model_segment,
            chili_method=args.chili_method,
            chili_threshold=args.chili_threshold,
            mesocarp_wrinkle_segment_threshold_mm=(
                args.mesocarp_wrinkle_segment_threshold_mm
            ),
        )

        for chili in chilis.items:
            row = {
                'id': chili.id,
                'name_root': name_image,
                'name_image': chili.name,
                'width_bbox': chili.width_bbox,
                'height_bbox': chili.height_bbox,
                'average_width': chili.width_fruit,
                'min_width': chili.width_fruit_stats['min'],
                'max_width': chili.width_fruit_stats['max'],
                'rednees': chili.redness,
                'area': chili.area_fruit,
                'mesocarp_area': chili.mesocarp_area,
                'placenta_area': chili.placenta_area,
                'flesh_placenta_ratio': chili.flesh_placenta_ratio,
                'min_mesocarp_thickness': chili.mesocarp_thickness_stats['min_mesocarp_thickness'],
                'average_mesocarp_thickness': chili.mesocarp_thickness_stats['avg_mesocarp_thickness'],
                'max_mesocarp_thickness': chili.mesocarp_thickness_stats['max_mesocarp_thickness'],
                'mesocarp_wrinkle_index': chili.mesocarp_wrinkle_index,
                'scale_ratio': chili.ratio,
                'scale_source': chili.scale_source,
            }
            rows.append(row)
        # results.append(chilis)
    results = pd.DataFrame(rows)
    
    if args.save_csv:
        results.to_csv('./runs/predict/output.csv')

    return results

def main():
    args = parse_args()

    print("Arguments:", args)
    results = inference(args)
    print(results)


if __name__ == "__main__":
    main()
