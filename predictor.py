import os
import matplotlib.pyplot as plt
import shutil
import glob
import argparse
import torch
import pandas as pd
from pathlib import Path

from src.detect import DetectModel
from src.segment import SegmentModel
from src.object import Fruits, Chili


WEIGHT_MODEL_DETECT = None
WEIGHT_MODEL_SEGMENT = './pretrain/yolov8_segment_v2.pt'
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description="Chili Pepper Segmentation")

    parser.add_argument("--input_path", type=str, required=True, help="Path or dir to input image")
    parser.add_argument("--use_model_det", action='store_true', help="Use model or threshold for get bbox of chili")
    parser.add_argument("--name_model_det", default='yolo', type=str, help="Name model object detection")
    parser.add_argument("--name_model_seg", default='yolo', type=str, help="Name model segmentation")
    parser.add_argument("--device", default=DEVICE, help="set device to run model")
    parser.add_argument("--weight_det", default=WEIGHT_MODEL_DETECT, type=str, required=False, help="Weight pretrain of model object detection for get bbox chili")
    parser.add_argument("--weight_seg", default=WEIGHT_MODEL_SEGMENT, type=str, required=False, help="Weight pretrain of model segmentation")
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
        "--wrinkle-segment-threshold-mm",
        default=5.0,
        type=positive_float,
        help="Outer-wall chord threshold in mm used by the wrinkle feature",
    )
    parser.add_argument("--save_csv", action="store_true", help="save output result")
    parser.add_argument("--save_img", action="store_true", help="save image output")

    return parser.parse_args()

def inference(args):
    if args.use_model_det:
        model_detect = DetectModel(model_name=args.name_model_det, weight_path=args.weight_det, device=args.device)
    else:
        model_detect = DetectModel(model_name='none')

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
            wrinkle_segment_threshold_mm=args.wrinkle_segment_threshold_mm,
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
                'min_thickness': chili.thickness_stats['min_thickness'],
                'average_thickness': chili.thickness_stats['avg_thickness'],
                'max_thickness': chili.thickness_stats['max_thickness'],
                'wrinkle': chili.wrinkle,
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
