#!/usr/bin/env python3
"""Evaluate fruit counts from both detector branches against LabelMe JSON."""

import argparse
import csv
import html
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detect import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_CONTAINMENT,
    DEFAULT_CROP_TOP,
    DEFAULT_INPUT_SIZE,
    DEFAULT_IOU,
    DetectModel,
)
from src.segment import SegmentModel  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402

LOGGER.setLevel("ERROR")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FIELDS = [
    "image", "json", "label_count",
    "yolov7_detector_count", "yolov7_pipeline_count",
    "yolov7_rejected_count", "yolov7_diff", "yolov7_abs_error",
    "yolov7_exact", "yolov7_rejection_reasons",
    "threshold_detector_count", "threshold_pipeline_count",
    "threshold_rejected_count", "threshold_diff", "threshold_abs_error",
    "threshold_exact", "threshold_rejection_reasons",
    "both_exact", "status", "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare YOLOv7 and threshold pipeline counts with JSON labels."
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=WORKSPACE_ROOT / "jeju_segmentation",
    )
    parser.add_argument(
        "--output-parent", type=Path, default=WORKSPACE_ROOT,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Resume an interrupted run from counts.partial.csv.",
    )
    parser.add_argument(
        "--weight-det", type=Path,
        default=PROJECT_ROOT / "weights" / "yolov7_detection.onnx",
    )
    parser.add_argument(
        "--weight-seg", type=Path,
        default=PROJECT_ROOT / "weights" / "yolov8_segment_v2.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--chili-method", choices=["threshold", "yolo"], default="threshold",
    )
    parser.add_argument("--chili-threshold", type=int, default=16)
    parser.add_argument(
        "--det-conf-thres", type=float, default=DEFAULT_CONFIDENCE,
    )
    parser.add_argument("--det-iou-thres", type=float, default=DEFAULT_IOU)
    parser.add_argument(
        "--det-containment-thres", type=float, default=DEFAULT_CONTAINMENT,
    )
    parser.add_argument(
        "--det-input-size", type=int, default=DEFAULT_INPUT_SIZE,
    )
    parser.add_argument("--det-crop-top", type=int, default=DEFAULT_CROP_TOP)
    parser.add_argument("--label-name", default="chili")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def make_output_dir(args):
    if args.resume_dir:
        output = args.resume_dir.expanduser().resolve()
        checkpoint = output / "counts.partial.csv"
        if not output.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError(
                "Resume directory must contain counts.partial.csv: " + str(output)
            )
        return output
    if args.output_dir:
        output = args.output_dir.expanduser().resolve()
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output = (
            args.output_parent.expanduser().resolve()
            / ("detection_count_evaluation_" + stamp)
        )
    if output.exists():
        raise FileExistsError("Output directory already exists: " + str(output))
    output.mkdir(parents=True)
    return output


def discover_pairs(dataset, limit):
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError("Dataset directory not found: " + str(dataset))
    images = sorted(
        p for p in dataset.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero")
        images = images[:limit]
    pairs, missing = [], []
    for image in images:
        label = image.with_suffix(".json")
        (pairs if label.is_file() else missing).append(
            (image, label) if label.is_file() else str(image)
        )
    return pairs, missing


def label_count(path, target):
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = target.strip().casefold()
    return sum(
        str(shape.get("label", "")).strip().casefold() == target
        for shape in payload.get("shapes", [])
    )


def crop_box(image, bbox, pad=50):
    height, width = image.shape[:2]
    x, y, box_width, box_height = map(
        lambda value: int(round(float(value))), bbox
    )
    return image[
        max(0, y - pad):min(height, y + box_height + pad),
        max(0, x - pad):min(width, x + box_width + pad),
    ]


def evaluate_backend(
    image_path, image_rgb, detector, segmenter, args, quiet_stream
):
    boxes = detector.predict(str(image_path))
    mode = "any_segment" if detector.backend == "threshold" else "detector_trusted"
    valid = 0
    reasons = Counter()
    for bbox in boxes:
        crop = crop_box(image_rgb, bbox)
        if crop.size == 0:
            reasons["empty_crop"] += 1
            continue
        try:
            with redirect_stdout(quiet_stream):
                result = segmenter.predict(
                    crop,
                    chili_method=args.chili_method,
                    chili_threshold=args.chili_threshold,
                    validation_mode=mode,
                )
            stats = result.get("stats", {})
            if stats.get("is_valid_chili", False):
                valid += 1
            else:
                reasons[stats.get("rejection_reason") or "invalid_candidate"] += 1
        except Exception as error:
            reasons["segment_error:" + type(error).__name__] += 1
    return {
        "detector_count": len(boxes),
        "pipeline_count": valid,
        "rejected_count": len(boxes) - valid,
        "reasons": dict(sorted(reasons.items())),
    }


def reasons_text(reasons):
    return ";".join(key + ":" + str(value) for key, value in reasons.items())


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def branch_summary(rows, prefix):
    valid = [row for row in rows if row["status"] == "OK"]
    differences = [int(row[prefix + "_diff"]) for row in valid]
    exact = sum(int(row[prefix + "_exact"]) for row in valid)
    labels = sum(int(row["label_count"]) for row in valid)
    detector = sum(int(row[prefix + "_detector_count"]) for row in valid)
    pipeline = sum(int(row[prefix + "_pipeline_count"]) for row in valid)
    rejected = sum(int(row[prefix + "_rejected_count"]) for row in valid)
    return {
        "images": len(valid),
        "exact_images": exact,
        "exact_rate": exact / len(valid) if valid else 0.0,
        "total_labels": labels,
        "total_detector_candidates": detector,
        "total_pipeline_fruits": pipeline,
        "total_rejected_candidates": rejected,
        "total_difference": pipeline - labels,
        "mae": (
            sum(abs(value) for value in differences) / len(differences)
            if differences else 0.0
        ),
        "rmse": (
            math.sqrt(
                sum(value * value for value in differences) / len(differences)
            ) if differences else 0.0
        ),
        "under_count_images": sum(value < 0 for value in differences),
        "over_count_images": sum(value > 0 for value in differences),
    }


def write_reports(output, rows, summary):
    yolo = summary["branches"]["yolov7"]
    threshold = summary["branches"]["threshold"]
    markdown = [
        "# Đánh giá số lượng quả trên toàn bộ dataset", "",
        "- Thời gian: " + summary["created_at"],
        "- Dataset: " + summary["dataset"],
        "- Số cặp ảnh/JSON: **" + str(summary["paired_images"]) + "**",
        "- Ảnh lỗi runtime: **" + str(summary["runtime_error_images"]) + "**",
        "",
        "| Nhánh | Label | Detector | Pipeline | Exact | MAE | RMSE | Under | Over |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in (("YOLOv7", yolo), ("Threshold", threshold)):
        markdown.append(
            "| {name} | {labels} | {detector} | {pipeline} | "
            "{exact}/{images} ({rate:.2%}) | {mae:.4f} | {rmse:.4f} | "
            "{under} | {over} |".format(
                name=name,
                labels=data["total_labels"],
                detector=data["total_detector_candidates"],
                pipeline=data["total_pipeline_fruits"],
                exact=data["exact_images"],
                images=data["images"],
                rate=data["exact_rate"],
                mae=data["mae"],
                rmse=data["rmse"],
                under=data["under_count_images"],
                over=data["over_count_images"],
            )
        )
    markdown += [
        "",
        "Chi tiết từng ảnh nằm trong counts.csv; lỗi nằm trong errors.csv.",
    ]
    (output / "report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(row["image"])),
                row["label_count"],
                row["yolov7_detector_count"],
                row["yolov7_pipeline_count"],
                row["yolov7_diff"],
                row["threshold_detector_count"],
                row["threshold_pipeline_count"],
                row["threshold_diff"],
                row["status"],
            )
        )
    document = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Detection count evaluation</title><style>
body{{font-family:Arial;margin:24px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{background:#f2f4f7;padding:14px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;margin-top:20px}}
th,td{{border:1px solid #ddd;padding:6px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}</style></head><body>
<h1>Đánh giá số lượng quả</h1><div class="cards">
<div class="card"><b>{pairs}</b><br>cặp ảnh/JSON</div>
<div class="card"><b>{yolo_rate:.2%}</b><br>YOLOv7 exact</div>
<div class="card"><b>{threshold_rate:.2%}</b><br>Threshold exact</div>
<div class="card"><b>{yolo_mae:.4f}</b><br>YOLOv7 MAE</div>
<div class="card"><b>{threshold_mae:.4f}</b><br>Threshold MAE</div>
</div><table><thead><tr><th>Ảnh</th><th>Label</th>
<th>YOLO det</th><th>YOLO final</th><th>YOLO diff</th>
<th>Threshold det</th><th>Threshold final</th><th>Threshold diff</th>
<th>Status</th></tr></thead><tbody>{rows}</tbody></table></body></html>""".format(
        pairs=summary["paired_images"],
        yolo_rate=yolo["exact_rate"],
        threshold_rate=threshold["exact_rate"],
        yolo_mae=yolo["mae"],
        threshold_mae=threshold["mae"],
        rows="".join(body_rows),
    )
    (output / "report.html").write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    started = time.perf_counter()
    output = make_output_dir(args)
    pairs, missing = discover_pairs(args.dataset, args.limit)
    if not pairs:
        raise RuntimeError("No matching image/JSON pairs found")

    yolo_detector = DetectModel(
        backend="yolov7_onnx",
        weight_path=args.weight_det,
        conf_threshold=args.det_conf_thres,
        iou_threshold=args.det_iou_thres,
        containment_threshold=args.det_containment_thres,
        input_size=args.det_input_size,
        crop_top=args.det_crop_top,
    )
    threshold_detector = DetectModel(
        backend="threshold", crop_top=args.det_crop_top,
    )
    segmenter = SegmentModel(
        weight_path=str(args.weight_seg),
        device=args.device,
        chili_method=args.chili_method,
        chili_threshold=args.chili_threshold,
    )

    rows, errors = [], []
    if args.resume_dir:
        with (output / "counts.partial.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        errors = [
            {
                "image": row["image"],
                "json": row["json"],
                "error": row["error"],
            }
            for row in rows
            if row["status"] == "ERROR"
        ]
    completed_images = {row["image"] for row in rows}

    with open(os.devnull, "w") as quiet:
        for index, (image_path, json_path) in enumerate(pairs, 1):
            if image_path.name in completed_images:
                continue
            row = {field: "" for field in FIELDS}
            row.update(
                image=image_path.name, json=json_path.name, status="OK"
            )
            try:
                expected = label_count(json_path, args.label_name)
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    raise ValueError("OpenCV could not read image")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results = {
                    "yolov7": evaluate_backend(
                        image_path, rgb, yolo_detector, segmenter, args, quiet
                    ),
                    "threshold": evaluate_backend(
                        image_path, rgb, threshold_detector, segmenter, args, quiet
                    ),
                }
                row["label_count"] = expected
                for prefix, result in results.items():
                    predicted = result["pipeline_count"]
                    difference = predicted - expected
                    row[prefix + "_detector_count"] = result["detector_count"]
                    row[prefix + "_pipeline_count"] = predicted
                    row[prefix + "_rejected_count"] = result["rejected_count"]
                    row[prefix + "_diff"] = difference
                    row[prefix + "_abs_error"] = abs(difference)
                    row[prefix + "_exact"] = int(difference == 0)
                    row[prefix + "_rejection_reasons"] = reasons_text(
                        result["reasons"]
                    )
                row["both_exact"] = int(
                    row["yolov7_exact"] and row["threshold_exact"]
                )
            except Exception as error:
                row["status"] = "ERROR"
                row["error"] = type(error).__name__ + ": " + str(error)
                errors.append({
                    "image": image_path.name,
                    "json": json_path.name,
                    "error": row["error"],
                })
            rows.append(row)
            write_rows(output / "counts.partial.csv", rows)
            if (
                index % max(1, args.progress_every) == 0
                or index == len(pairs)
            ):
                print(
                    "[{}/{}] {} GT={} YOLO={}/{} THR={}/{} elapsed={:.1f}s".format(
                        index, len(pairs), image_path.name, row["label_count"],
                        row["yolov7_detector_count"],
                        row["yolov7_pipeline_count"],
                        row["threshold_detector_count"],
                        row["threshold_pipeline_count"],
                        time.perf_counter() - started,
                    ),
                    flush=True,
                )

    write_rows(output / "counts.csv", rows)
    (output / "counts.partial.csv").unlink(missing_ok=True)
    with (output / "errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["image", "json", "error"]
        )
        writer.writeheader()
        writer.writerows(errors)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": str(args.dataset.expanduser().resolve()),
        "output_directory": str(output),
        "paired_images": len(pairs),
        "missing_json_count": len(missing),
        "missing_json_images": missing,
        "runtime_error_images": len(errors),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "config": {
            "label_name": args.label_name,
            "chili_method": args.chili_method,
            "chili_threshold": args.chili_threshold,
            "yolov7_confidence": args.det_conf_thres,
            "yolov7_iou": args.det_iou_thres,
            "yolov7_containment_iom": args.det_containment_thres,
            "detector_input_size": args.det_input_size,
            "detector_crop_top": args.det_crop_top,
            "device": str(args.device),
            "detection_weight": str(args.weight_det.resolve()),
            "segmentation_weight": str(args.weight_seg.resolve()),
        },
        "branches": {
            "yolov7": branch_summary(rows, "yolov7"),
            "threshold": branch_summary(rows, "threshold"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_reports(output, rows, summary)
    print("Results: " + str(output))
    print(json.dumps(summary["branches"], indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
