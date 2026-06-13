import argparse
from pathlib import Path

import pandas as pd

from src.debug import FEATURE_COLUMNS, save_chili_debug


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SEGMENT_WEIGHT = "./pretrain/yolov8_segment_v1.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run chili inference and save evaluation/debug artifacts."
    )
    parser.add_argument("--input_path", required=True, help="Input image or folder")
    parser.add_argument(
        "--output_dir",
        default="./runs/evaluation",
        help="Directory for images and features.csv",
    )
    parser.add_argument(
        "--weight_seg",
        default=DEFAULT_SEGMENT_WEIGHT,
        help="YOLO segmentation weights",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device; defaults to CUDA 0 when available, otherwise CPU",
    )
    parser.add_argument(
        "--use_model_det",
        action="store_true",
        help="Use YOLO detection instead of threshold detection",
    )
    parser.add_argument("--weight_det", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on the first image error",
    )
    return parser.parse_args()


def list_images(input_path, recursive=False):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_models(args):
    import torch

    from src.detect import DetectModel
    from src.segment import SegmentModel

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    if args.use_model_det:
        if not args.weight_det:
            raise ValueError("--weight_det is required with --use_model_det")
        model_detect = DetectModel(
            model_name="yolo",
            weight_path=args.weight_det,
            device=device,
        )
    else:
        model_detect = DetectModel(model_name="none")

    model_segment = SegmentModel(
        weight_path=args.weight_seg,
        model_name="yolo",
        device=device,
    )
    return model_detect, model_segment


def create_fruits(image_path, model_detect, model_segment):
    from src.object import Fruits

    return Fruits(
        str(image_path),
        model_detect=model_detect,
        model_segment=model_segment,
    )


def image_output_dir(output_root, input_path, image_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return output_root / image_path.stem

    relative_path = image_path.relative_to(input_path).with_suffix("")
    return output_root / relative_path


def run_evaluation(args):
    image_paths = list_images(args.input_path, args.recursive)
    if not image_paths:
        raise ValueError(f"No supported images found in: {args.input_path}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model_detect, model_segment = build_models(args)

    rows = []
    errors = []
    for image_index, image_path in enumerate(image_paths, start=1):
        print(f"[{image_index}/{len(image_paths)}] {image_path}")
        image_output = image_output_dir(
            output_root,
            args.input_path,
            image_path,
        )
        try:
            fruits = create_fruits(image_path, model_detect, model_segment)
            for chili in fruits.items:
                chili_output = image_output / f"chili_{chili.id:03d}"
                rows.append(
                    save_chili_debug(
                        chili,
                        source_image=image_path,
                        output_dir=chili_output,
                    )
                )
        except Exception as exc:
            errors.append({
                "source_image": str(image_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise

    features_path = output_root / "features.csv"
    results = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    results.to_csv(features_path, index=False)

    errors_path = output_root / "errors.csv"
    pd.DataFrame(
        errors,
        columns=["source_image", "error_type", "error"],
    ).to_csv(errors_path, index=False)

    print(f"Saved {len(rows)} chili rows to {features_path}")
    if errors:
        print(f"Saved {len(errors)} image errors to {errors_path}")
    return results


def main():
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
