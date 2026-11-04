"""Presentation helpers for the HTML demo; measurements stay in src.pipeline."""
import math
import os
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from src.progress import notify

import cv2
import numpy as np
import pandas as pd

from src.detect import DEFAULT_CROP_TOP, get_mask as get_detection_mask
from src.object import CLASS_COLORS, CLASS_NAMES, draw_full_mask_with_labels
from src.pipeline import (PipelineConfig, build_models, run_image, feature_columns,
                          feature_row, create_manifest, image_diagnostics, write_exports)

BASE_DIR = Path(__file__).resolve().parents[1]
DETECTION_CROP_TOP = DEFAULT_CROP_TOP
FRUIT_MASK_COLOR = (155, 0, 155)
EXTRA_MASKS = {"pedicel": (255, 165, 0), "placenta": (0, 215, 255)}
FEATURE_COLUMNS = feature_columns()


def make_config(options):
    import torch
    return PipelineConfig(
        detector_backend=options.detector_backend,
        weight_det=os.getenv("CHILI_DETECTION_WEIGHT", str(BASE_DIR / "weights/yolov7_detection.onnx")),
        weight_seg=os.getenv("CHILI_SEGMENT_WEIGHT", str(BASE_DIR / "weights/yolov11m_segment_final.pt")),
        device="0" if torch.cuda.is_available() else "cpu",
        fruit_threshold_offset=options.fruit_threshold_offset,
        wrinkle_tau_mm=options.wrinkle_tau_mm, seg_imgsz=options.seg_imgsz,
        det_crop_top=options.crop_top, det_conf=options.det_conf,
        det_iou=options.det_iou, det_containment=options.det_containment,
    )


@lru_cache(maxsize=4)
def _get_models(config):
    # Only the sequential inference worker calls the shared models.
    for name, path in (("segmentation", config.weight_seg), ("detection", config.weight_det)):
        if name == "detection" and config.detector_backend == "threshold":
            continue
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} weights ({Path(path).name}). Place the checkpoint in weights/ or configure its environment variable.")
    return build_models(config)


def read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("The image cannot be decoded. Please choose a valid image file.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_image(path, crop_top):
    image = read_rgb(path)
    height, width = image.shape[:2]
    if height <= crop_top:
        raise ValueError(f"Image height is {height}px, but the top crop is {crop_top}px. Set top crop to 0 for images without the dataset header.")
    return width, height


def _draw_detection_boxes(image_rgb, fruits, label_prefix="Chili", index_offset=0):
    """Draw the detector bbox and the padded crop used by Chili.get_img()."""
    output = image_rgb.copy()
    height, width = output.shape[:2]

    for index, bbox in enumerate(fruits.box_cut):
        x, y, box_width, box_height = [int(value) for value in bbox[:4]]
        x2, y2 = x + box_width, y + box_height

        cv2.rectangle(output, (x, y), (x2, y2), (0, 255, 0), 4)

        pad = 50
        crop_x1, crop_y1 = max(0, x - pad), max(0, y - pad)
        crop_x2, crop_y2 = min(width, x2 + pad), min(height, y2 + pad)
        cv2.rectangle(
            output,
            (crop_x1, crop_y1),
            (crop_x2, crop_y2),
            (0, 220, 255),
            2,
        )

        label_y = max(30, y - 12)
        cv2.putText(
            output,
            f"{label_prefix} {index + index_offset}",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"{label_prefix} {index + index_offset}",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 150, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def _draw_roi_guide(image_rgb, crop_top=DETECTION_CROP_TOP):
    """Visualize the fixed top crop performed by DetectModel.predict()."""
    output = image_rgb.copy()
    height, width = output.shape[:2]
    crop_y = min(crop_top, height)

    if crop_y > 0:
        dark = output[:crop_y].astype(np.float32) * 0.25
        output[:crop_y] = dark.astype(np.uint8)

    if crop_y < height:
        cv2.line(output, (0, crop_y), (width - 1, crop_y), (255, 220, 0), 5)
        cv2.putText(
            output,
            f"Detection ROI starts at y={crop_top}",
            (20, min(height - 15, crop_y + 45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 220, 0),
            3,
            cv2.LINE_AA,
        )

    return output


def _overlay_mask(image_rgb, mask, color):
    """Render one mask over the complete fruit crop, including empty masks."""
    output = (image_rgb.astype(np.float32) * 0.22).astype(np.uint8)
    mask_bool = mask > 0
    if not np.any(mask_bool):
        cv2.putText(
            output,
            "NO PIXELS DETECTED",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    color_array = np.asarray(color, dtype=np.float32)
    output[mask_bool] = (
        image_rgb[mask_bool].astype(np.float32) * 0.25 + color_array * 0.75
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def _restore_original_orientation(array, rotation):
    """Undo Chili orientation normalization for projection onto the source image."""
    if rotation == "90_cw":
        return cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "90_ccw":
        return cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "180":
        return cv2.rotate(array, cv2.ROTATE_180)
    return array.copy()


def _build_full_segmentation(image_rgb, fruits):
    """Project every per-fruit class mask back onto the complete source image."""
    output = image_rgb.copy()
    mask_layers = [
        ("mask_fruit", FRUIT_MASK_COLOR),
        ("mask_endocarp", CLASS_COLORS[3]),
        ("mask_pcp", CLASS_COLORS[2]),
        ("mask_outer", CLASS_COLORS[1]),
        ("mask_seed", CLASS_COLORS[0]),
    ]

    for chili in fruits.items:
        x1, x2, y1, y2 = [int(value) for value in chili.position]
        crop_height, crop_width = y2 - y1, x2 - x1
        if crop_height <= 0 or crop_width <= 0:
            continue

        region = output[y1:y2, x1:x2]
        for attribute, color in mask_layers:
            mask = getattr(chili, attribute, None)
            if mask is None or not np.any(mask > 0):
                continue

            restored = _restore_original_orientation(
                mask,
                getattr(chili, "rotation", "none"),
            )
            if restored.shape != (crop_height, crop_width):
                restored = cv2.resize(
                    restored,
                    (crop_width, crop_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            active = restored > 0
            color_array = np.asarray(color, dtype=np.float32)
            region[active] = (
                region[active].astype(np.float32) * 0.45 + color_array * 0.55
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                restored.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(region, contours, -1, color, 2)

        cv2.rectangle(region, (0, 0), (crop_width - 1, crop_height - 1), (255, 255, 255), 2)
        cv2.putText(
            region,
            f"Chili {chili.id}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            region,
            f"Chili {chili.id}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def _fit_on_canvas(image, width, height, background=(245, 245, 245)):
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    source_height, source_width = image.shape[:2]
    if source_height == 0 or source_width == 0:
        return canvas

    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def _format_metric(value, digits=2):
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _build_feature_card(raw_crop, segmented_crop, chili):
    """Create one comparable original/segmentation/metrics image per fruit."""
    card_height = 620
    header_height = 50
    image_width = 240
    panel_width = 520
    content_height = card_height - header_height
    card_width = image_width * 2 + panel_width

    card = np.full((card_height, card_width, 3), 248, dtype=np.uint8)
    card[:header_height] = (36, 42, 48)

    cv2.putText(
        card,
        f"CHILI {chili.id}  |  {chili.name}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    original_canvas = _fit_on_canvas(raw_crop, image_width, content_height)
    segment_canvas = _fit_on_canvas(segmented_crop, image_width, content_height)
    card[header_height:, :image_width] = original_canvas
    card[header_height:, image_width:image_width * 2] = segment_canvas
    cv2.line(card, (image_width, header_height), (image_width, card_height), (180, 180, 180), 2)
    cv2.line(card, (image_width * 2, header_height), (image_width * 2, card_height), (180, 180, 180), 2)

    cv2.putText(card, "ORIGINAL", (65, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(card, "SEGMENT", (image_width + 62, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

    widths = chili.width_fruit_stats
    thickness = chili.mesocarp_thickness_stats
    seeds = chili.seed_stats
    estimated_scale = chili.scale_source == "dataset_mean_fallback"
    unit = "mm"
    area_unit = "mm2"
    scale_unit = "mm/pixel (dataset mean)" if estimated_scale else "mm/pixel"
    metrics = [
        f"Scale: {_format_metric(chili.ratio, 6)}  ({scale_unit})",
        f"BBox W x H: {_format_metric(chili.width_bbox)} x {_format_metric(chili.height_bbox)} {unit}",
        f"FW_min / FW_mean / FW_max: {_format_metric(widths['min'])} / {_format_metric(widths['avg'])} / {_format_metric(widths['max'])} {unit}",
        f"Fruit Length (FL): {_format_metric(chili.length_fruit)} {unit}",
        f"Fruit Area (FA): {_format_metric(chili.area_fruit)} {area_unit}",
        f"Mesocarp Area (MA): {_format_metric(chili.mesocarp_area)} {area_unit}",
        f"Placenta Area (PA): {_format_metric(chili.placenta_area)} {area_unit}",
        f"Mesocarp Area Fraction (MAF): {_format_metric(chili.mesocarp_area_fraction, 4)}",
        f"MT_min / MT_mean / MT_max: {_format_metric(thickness['min_mesocarp_thickness'])} / {_format_metric(thickness['avg_mesocarp_thickness'])} / {_format_metric(thickness['max_mesocarp_thickness'])} {unit}",
        f"Seed Count (SC): {_format_metric(chili.number_seeds)}",
        f"Total Seed Area (SA): {_format_metric(seeds['sum'])} {area_unit}",
        f"Redness / uniformity: {_format_metric(chili.redness, 4)} / {_format_metric(chili.color_uniformity, 4)}",
        f"Mesocarp Wrinkle Index (MWI): {_format_metric(chili.mesocarp_wrinkle_index, 5)}",
        f"Fruit Curvature Index (FCI): {_format_metric(chili.curvature_index, 4)}  (0 = straight; higher = more curved)",
        f"Seed-to-Cavity Area Ratio (SCR): {_format_metric(chili.seed_density, 4)}",
        f"D_L / D_R / D_mean: {_format_metric(chili.deviation_stats['D_L'], 4)} / {_format_metric(chili.deviation_stats['D_R'], 4)} / {_format_metric(chili.deviation_stats['D_mean'], 4)}",
    ]

    text_x = image_width * 2 + 18
    cv2.putText(card, "COMPARISON FEATURES", (text_x, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    for index, line in enumerate(metrics):
        text_y = 105 + index * 30
        cv2.putText(
            card,
            line,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )

    if estimated_scale:
        cv2.putText(
            card,
            "* QR unavailable: using estimated dataset-mean scale",
            (text_x, card_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (180, 60, 20),
            1,
            cv2.LINE_AA,
        )

    return card


def _build_comparison_gallery(fruits, image_rgb):
    gallery = []
    for chili in fruits.items:
        x1, x2, y1, y2 = [int(value) for value in chili.position]
        raw_crop = image_rgb[y1:y2, x1:x2]
        segmented = draw_full_mask_with_labels(chili.img, chili.polygons)
        card = _build_feature_card(raw_crop, segmented, chili)
        gallery.append((card, f"Chili {chili.id} · visual feature comparison"))
    return gallery


def _build_mask_tile(image, label, color, tile_width, tile_content_height):
    """Build one labeled original/class tile while preserving the full fruit."""
    header_height = 42
    tile = np.full(
        (header_height + tile_content_height, tile_width, 3),
        245,
        dtype=np.uint8,
    )
    tile[:header_height] = (36, 42, 48)
    tile[header_height:] = _fit_on_canvas(
        image,
        tile_width,
        tile_content_height,
    )
    cv2.rectangle(tile, (10, 10), (30, 30), color, -1)
    cv2.rectangle(tile, (10, 10), (30, 30), (255, 255, 255), 1)
    cv2.putText(
        tile,
        label.upper(),
        (39, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _build_class_mask_sheet(chili, mask_attributes):
    """Group original plus all single-class views in one adaptive sheet."""
    tile_content_height = 360
    image_height, image_width = chili.img.shape[:2]
    aspect_ratio = image_width / max(image_height, 1)
    tile_width = int(
        np.clip(round(tile_content_height * aspect_ratio), 170, 420)
    )

    gap = 12
    max_single_row_width = 1900
    number_of_tiles = 1 + len(mask_attributes)
    one_row_width = number_of_tiles * tile_width + (number_of_tiles + 1) * gap
    columns = number_of_tiles if one_row_width <= max_single_row_width else 4
    rows = math.ceil(number_of_tiles / columns)

    tiles = [
        _build_mask_tile(
            chili.img,
            "original",
            (120, 120, 120),
            tile_width,
            tile_content_height,
        )
    ]

    for label, attribute, color in mask_attributes:
        mask = getattr(chili, attribute, None)
        if mask is None:
            mask = np.zeros(chili.img.shape[:2], dtype=np.uint8)
        overlay = _overlay_mask(chili.img, mask, color)
        area_px = int(np.count_nonzero(mask))
        status = "empty" if area_px == 0 else f"{area_px:,} px"
        tiles.append(
            _build_mask_tile(
                overlay,
                f"{label} | {status}",
                color,
                tile_width,
                tile_content_height,
            )
        )

    tile_height = tiles[0].shape[0]
    sheet_header_height = 48
    sheet_width = columns * tile_width + (columns + 1) * gap
    sheet_height = (
        sheet_header_height + rows * tile_height + (rows + 1) * gap
    )
    sheet = np.full((sheet_height, sheet_width, 3), 232, dtype=np.uint8)
    sheet[:sheet_header_height] = (24, 28, 33)
    cv2.putText(
        sheet,
        f"CHILI {chili.id}  |  CLASS MASKS  |  LAYOUT {columns}x{rows}",
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        x1 = gap + column * (tile_width + gap)
        y1 = sheet_header_height + gap + row * (tile_height + gap)
        sheet[y1:y1 + tile_height, x1:x1 + tile_width] = tile

    return sheet, columns, rows


def _build_crop_gallery(fruits, image_rgb):
    gallery = []
    for chili in fruits.items:
        x1, x2, y1, y2 = chili.position
        raw_crop = image_rgb[y1:y2, x1:x2]
        gallery.append((raw_crop, f"Chili {chili.id} · raw padded crop"))
        gallery.append((chili.img, f"Chili {chili.id} · normalized orientation"))
    return gallery


def _build_segmentation_gallery(fruits):
    gallery = []
    for chili in fruits.items:
        rendered = draw_full_mask_with_labels(chili.img, chili.polygons)
        detected_classes = sorted(
            {
                polygon.get("class_name", CLASS_NAMES.get(polygon["class_id"], "unknown"))
                for polygon in chili.polygons
            }
        )
        class_text = ", ".join(detected_classes) if detected_classes else "no masks"
        gallery.append((rendered, f"Chili {chili.id} · {class_text}"))
    return gallery


def _build_class_mask_gallery(fruits):
    gallery = []
    mask_attributes = [
        ("seed", "mask_seed", CLASS_COLORS[0]),
        ("outer", "mask_outer", CLASS_COLORS[1]),
        ("PCP", "mask_pcp", CLASS_COLORS[2]),
        ("endocarp", "mask_endocarp", CLASS_COLORS[3]),
        ("whole-fruit mask (threshold)", "mask_fruit", FRUIT_MASK_COLOR),
        ("pedicel", "mask_pedicel", EXTRA_MASKS["pedicel"]),
        ("placenta", "mask_placenta", EXTRA_MASKS["placenta"]),
    ]

    for chili in fruits.items:
        sheet, columns, rows = _build_class_mask_sheet(chili, mask_attributes)
        gallery.append(
            (sheet, f"Chili {chili.id} · adaptive class-mask layout {columns}×{rows}")
        )
    return gallery



def extract_chili_features(image_path):
    """Compatibility helper: retain the original single-dataframe interface."""
    if image_path is None or not Path(image_path).is_file():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    from types import SimpleNamespace
    config = make_config(SimpleNamespace(detector_backend="yolov7_onnx", fruit_threshold_offset=16,
        wrinkle_tau_mm=5.0, seg_imgsz=640, crop_top=540, det_conf=.70, det_iou=.30, det_containment=.60))
    return pd.DataFrame([feature_row(c) for c in run_image(image_path, config, _get_models(config)).items], columns=FEATURE_COLUMNS)


def process_image(image_path, config, output_dir, source, on_progress=None):
    """Publish immutable stage artifacts while the unchanged measurements run."""
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    saved = {}
    candidate_artifacts = {}
    accepted = []

    def save(image, name, caption):
        if name not in saved:
            path = output_dir / "images" / f"{name}.png"
            temporary = path.with_name(path.stem + ".part.png")
            pixels = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
            if not cv2.imwrite(str(temporary), pixels):
                raise RuntimeError("A result image could not be saved.")
            os.replace(temporary, path)
            saved[name] = {"url": f"images/{name}.png", "caption": caption}
        return dict(saved[name])

    def report(event):
        # Subjects contain arrays and models; only artifact references and plain
        # metadata cross the API boundary. Serialization happens synchronously.
        public = {key: value for key, value in event.items() if key not in {"subject", "boxes"}}
        subject = event.get("subject")
        node, state = event.get("node"), event["state"]
        index = event.get("candidate_index")
        outputs = {"images": [], "details": {}}
        if node == "scale" and state == "completed":
            outputs["details"] = {"mm_per_pixel": float(subject.ratio), "source": subject.scale_source,
                                  "warning": subject.scale_warning}
        elif node == "detection" and state == "completed":
            boxes = [[int(v) for v in box] for box in event["boxes"]]
            outputs["images"] = [save(_draw_detection_boxes(image_rgb, SimpleNamespace(box_cut=boxes), label_prefix="Candidate", index_offset=1),
                "detected-candidates", "Detection candidates · before segmentation validation")]
            outputs["details"] = {"candidate_count": len(boxes), "boxes_xywh": boxes}
        elif index is not None and state == "completed" and subject is not None:
            artifacts = candidate_artifacts.setdefault(index, {})
            name = f"candidate-{index}"
            if node == "crop":
                artifacts["raw"] = save(subject.img, name + "-raw", f"Candidate {index + 1} · padded crop")
                outputs["images"] = [artifacts["raw"]]
                outputs["details"] = {"bbox_xywh": [int(v) for v in subject.bbox], "padding_px": 50}
            elif node == "segmentation":
                outputs["images"] = [
                    save(subject.img, name + "-oriented", f"Candidate {index + 1} · orientation-normalized input"),
                    save(draw_full_mask_with_labels(subject.img, subject.rseg["polygons"]),
                         name + "-model-overlay", f"Candidate {index + 1} · model segmentation")]
                outputs["details"] = {"rotation": subject.rotation,
                    "model_detections": len(subject.rseg.get("detections", [])),
                    "whole_fruit_mask_source": "red_channel_threshold"}
            elif node == "masks":
                artifacts["normalized"] = save(subject.img, name + "-normalized", f"Chili {subject.id} · normalized orientation")
                segment, caption = _build_segmentation_gallery(SimpleNamespace(items=[subject]))[0]
                artifacts["segmentation"] = save(segment, name + "-segmentation", caption)
                sheet, caption = _build_class_mask_gallery(SimpleNamespace(items=[subject]))[0]
                artifacts["class_masks"] = save(sheet, name + "-masks", caption)
                outputs["images"] = [artifacts["segmentation"], artifacts["class_masks"]]
                outputs["details"] = {"mask_classes": ["seed", "outer", "PCP", "endocarp", "whole fruit", "pedicel", "placenta"]}
            elif node == "traits":
                card, caption = _build_comparison_gallery(SimpleNamespace(items=[subject]), image_rgb)[0]
                artifacts["comparison"] = save(card, name + "-comparison", caption)
                outputs["images"] = [artifacts["comparison"]]
                frame = pd.DataFrame([feature_row(subject)], columns=FEATURE_COLUMNS)
                outputs["features"] = {"columns": FEATURE_COLUMNS,
                    "rows": frame.astype(object).where(pd.notna(frame), None).values.tolist()}
                outputs["details"] = {"groups": ["geometry", "seeds", "shape", "color"]}
        if event.get("kind") == "candidate" and state == "completed":
            accepted.append(index)
        if outputs["images"] or outputs["details"] or "features" in outputs:
            public["outputs"] = outputs
        if on_progress is not None:
            on_progress(public)

    notify(on_progress, "input", "running")
    image_rgb = read_rgb(image_path)
    height, width = image_rgb.shape[:2]
    validate_image(image_path, config.det_crop_top)
    roi = image_rgb[config.det_crop_top:].copy()
    images = {
        "original": save(image_rgb, "original", "Original input image"),
        "roi_guide": save(_draw_roi_guide(image_rgb, config.det_crop_top), "roi-guide", "Darkened header is excluded from detection"),
        "detection_roi": save(roi, "detection-roi", "Image passed to the detector"),
        "threshold_mask": save(get_detection_mask(roi, threshold=22, use_closing=True, use_contour=False), "threshold-mask", "Reference red-channel threshold mask"),
    }
    notify(on_progress, "input", "completed", outputs={"images": list(images.values()),
        "details": {"filename": image_path.name, "width": width, "height": height, "top_crop_px": config.det_crop_top}})
    notify(on_progress, "models", "running")
    cache_hits = _get_models.cache_info().hits
    models = _get_models(config)
    notify(on_progress, "models", "completed", outputs={"details": {
        "cache_hit": _get_models.cache_info().hits > cache_hits,
        "detector": config.detector_backend, "segmentation": Path(config.weight_seg).name, "device": config.device}})
    started_at = time.perf_counter()
    fruits = run_image(image_path, config, models, on_progress=report)
    elapsed = time.perf_counter() - started_at

    notify(on_progress, "results", "running")
    images["detection"] = save(_draw_detection_boxes(image_rgb, fruits), "detection", "Accepted fruit · green: detection box · yellow: padded crop")
    images["full_segmentation"] = save(_build_full_segmentation(image_rgb, fruits), "full-segmentation", "Segmentation projected onto the source image")
    galleries = {"crops": [], "segmentation": [], "class_masks": [], "comparison": []}
    for index in accepted:
        artifacts = candidate_artifacts[index]
        galleries["crops"].extend([artifacts["raw"], artifacts["normalized"]])
        for key in ("segmentation", "class_masks", "comparison"):
            galleries[key].append(artifacts[key])
    features = pd.DataFrame([feature_row(c) for c in fruits.items], columns=FEATURE_COLUMNS)
    feature_table = {"columns": FEATURE_COLUMNS,
        "rows": features.astype(object).where(pd.notna(features), None).values.tolist()}
    notify(on_progress, "results", "completed", outputs={"images": [images["full_segmentation"], images["detection"]],
        "features": feature_table, "details": {"accepted_fruit": fruits.num_fruit,
        "rejected_candidates": len(fruits.candidate_diagnostics["rejected"])}})

    notify(on_progress, "export", "running")
    manifest = create_manifest(config, [image_path])
    entry = image_diagnostics(fruits)
    entry["elapsed_seconds"] = elapsed
    manifest["images"].append(entry)
    csv_path, json_path = write_exports(output_dir, features, manifest)
    temporary_zip = output_dir / "chili_results.zip.part"
    with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (csv_path, json_path):
            archive.write(path, path.name)
    os.replace(temporary_zip, output_dir / "chili_results.zip")
    downloads = {"zip": "chili_results.zip", "csv": "output.csv", "manifest": "manifest.json"}
    notify(on_progress, "export", "completed", outputs={"downloads": downloads,
        "details": {"rows": len(features), "columns": len(FEATURE_COLUMNS)}})
    warnings = []
    if fruits.num_fruit == 0:
        warnings.append("No fruit detected. Check the detection ROI, detector settings and segmentation validation.")
    if fruits.scale_warning:
        warnings.append(str(fruits.scale_warning))
    legend = [(CLASS_NAMES[i], CLASS_COLORS[i]) for i in range(4)] + [
        ("Whole-fruit mask (threshold)", FRUIT_MASK_COLOR), *EXTRA_MASKS.items()]
    return {
        "summary": {"source": source, "filename": image_path.name, "width": width, "height": height,
                    "fruit_count": fruits.num_fruit, "elapsed_seconds": elapsed,
                    "scale_ratio": float(fruits.ratio), "scale_source": fruits.scale_source,
                    "device": config.device, "configuration": config.effective_settings()},
        "warnings": warnings, "images": images, "galleries": galleries,
        "legend": [{"label": label, "color": list(color)} for label, color in legend],
        "features": feature_table, "downloads": downloads,
    }
