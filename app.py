import math
import os
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch

from src.detect import (
    DEFAULT_CONFIDENCE,
    DEFAULT_CROP_TOP,
    DEFAULT_IOU,
    DetectModel,
    get_mask as get_detection_mask,
)
from src.object import CLASS_COLORS, CLASS_NAMES, Fruits, draw_full_mask_with_labels
from src.segment import SegmentModel


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DETECTION_CROP_TOP = DEFAULT_CROP_TOP
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WEIGHT_MODEL_DETECT = Path(
    os.getenv(
        "CHILI_DETECTION_WEIGHT",
        str(BASE_DIR / "weights" / "yolov7_detection.onnx"),
    )
)
WEIGHT_MODEL_SEGMENT = Path(
    os.getenv(
        "CHILI_SEGMENT_WEIGHT",
        str(BASE_DIR / "weights" / "yolov8_segment_v2.pt"),
    )
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SAMPLE_IMAGES = sorted(
    path.name
    for path in DATA_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
) if DATA_DIR.exists() else []

FEATURE_COLUMNS = [
    "ID",
    "Name",
    "Width BBox",
    "Height BBox",
    "Width Min",
    "Width Max",
    "Average Width",
    "Length",
    "Width (9 segments)",
    "Length (9 segments)",
    "Area",
    "Mesocarp Area",
    "Fruit Body Area",
    "Placenta Area",
    "Min Mesocarp Thickness",
    "Average Mesocarp Thickness",
    "Max Mesocarp Thickness",
    "Flesh-to-Placenta Ratio",
    "Seed Count",
    "Seed Area Sum",
    "Seed Area Min",
    "Seed Area Max",
    "Seed Area Average",
    "Seed Area Std",
    "Redness",
    "Color Uniformity",
    "Mesocarp Wrinkle Index",
    "Curvature Index",
    "Seed Density",
    "Asymmetry Index",
    "Scale Ratio",
    "Scale Source",
]

EXTRA_MASKS = {
    "pedicel": (255, 165, 0),
    "placenta": (0, 215, 255),
}


def _color_legend_html(items):
    legend_items = []
    for label, color in items:
        red, green, blue = color
        legend_items.append(
            "<div style='display:flex;align-items:center;gap:8px;"
            "padding:6px 10px;border:1px solid #d1d5db;border-radius:8px;'>"
            f"<span style='width:22px;height:22px;background:rgb({red},{green},{blue});"
            "border:1px solid #fff;box-shadow:0 0 0 1px #777;display:inline-block;'></span>"
            f"<code>{label}</code></div>"
        )
    return (
        "<div style='display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 14px 0;'>"
        + "".join(legend_items)
        + "</div>"
    )


if not WEIGHT_MODEL_DETECT.exists():
    raise FileNotFoundError(
        f"YOLOv7 ONNX detection weight not found: {WEIGHT_MODEL_DETECT}"
    )
if not WEIGHT_MODEL_SEGMENT.exists():
    raise FileNotFoundError(
        f"Segmentation weight not found: {WEIGHT_MODEL_SEGMENT}"
    )

print(f"Loading pipeline models on device {DEVICE}...")
model_detect_onnx = DetectModel(
    backend="yolov7_onnx",
    weight_path=WEIGHT_MODEL_DETECT,
)
model_detect_threshold = DetectModel(backend="threshold")
# Backward-compatible default used by extract_chili_features().
model_detect = model_detect_onnx
model_segment = SegmentModel(
    weight_path=str(WEIGHT_MODEL_SEGMENT),
    model_name="yolo",
    device=DEVICE,
)
print("Pipeline models loaded successfully.")


def _resolve_input_path(upload_path, sample_name):
    """Use an uploaded image first, otherwise safely resolve a data sample."""
    if upload_path:
        path = Path(upload_path)
        if not path.is_file():
            raise gr.Error("Ảnh upload không còn tồn tại hoặc không đọc được.")
        return path, "upload"

    if not sample_name:
        raise gr.Error("Hãy upload ảnh hoặc chọn một ảnh trong thư mục data.")

    path = (DATA_DIR / sample_name).resolve()
    if path.parent != DATA_DIR.resolve() or not path.is_file():
        raise gr.Error("Ảnh được chọn không hợp lệ.")
    return path, "data"


def _read_rgb(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise gr.Error(f"OpenCV không đọc được ảnh: {image_path.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_detection_boxes(image_rgb, fruits):
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
            f"Chili {index}",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"Chili {index}",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 150, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def _draw_roi_guide(image_rgb):
    """Visualize the fixed top crop performed by DetectModel.predict()."""
    output = image_rgb.copy()
    height, width = output.shape[:2]
    crop_y = min(DETECTION_CROP_TOP, height)

    if crop_y > 0:
        dark = output[:crop_y].astype(np.float32) * 0.25
        output[:crop_y] = dark.astype(np.uint8)

    if crop_y < height:
        cv2.line(output, (0, crop_y), (width - 1, crop_y), (255, 220, 0), 5)
        cv2.putText(
            output,
            f"Detection ROI starts at y={DETECTION_CROP_TOP}",
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
        ("mask_chili", CLASS_COLORS[4]),
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
        f"Width min / avg / max: {_format_metric(widths['min'])} / {_format_metric(widths['avg'])} / {_format_metric(widths['max'])} {unit}",
        f"Length: {_format_metric(chili.length_fruit)} {unit}",
        f"Whole fruit area: {_format_metric(chili.area_fruit)} {area_unit}",
        f"Mesocarp area: {_format_metric(chili.mesocarp_area)} {area_unit}",
        f"Placenta area: {_format_metric(chili.placenta_area)} {area_unit}",
        f"Flesh / placenta: {_format_metric(chili.flesh_placenta_ratio)}",
        f"Mesocarp thickness min / avg / max: {_format_metric(thickness['min_mesocarp_thickness'])} / {_format_metric(thickness['avg_mesocarp_thickness'])} / {_format_metric(thickness['max_mesocarp_thickness'])} {unit}",
        f"Seed count: {_format_metric(chili.number_seeds)}",
        f"Seed area total: {_format_metric(seeds['sum'])} {area_unit}",
        f"Redness / uniformity: {_format_metric(chili.redness, 4)} / {_format_metric(chili.color_uniformity, 4)}",
        f"Mesocarp wrinkle index: {_format_metric(chili.mesocarp_wrinkle_index, 5)}",
        f"Curvature index: {_format_metric(chili.curvature_index, 4)}  (1 = straight)",
        f"Seed density: {_format_metric(chili.seed_density, 4)}",
        f"Asymmetry index: {_format_metric(chili.asymmetry_index, 4)}",
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
        ("chili", "mask_chili", CLASS_COLORS[4]),
        ("pedicel", "mask_pedicel", EXTRA_MASKS["pedicel"]),
        ("placenta", "mask_placenta", EXTRA_MASKS["placenta"]),
    ]

    for chili in fruits.items:
        sheet, columns, rows = _build_class_mask_sheet(chili, mask_attributes)
        gallery.append(
            (sheet, f"Chili {chili.id} · adaptive class-mask layout {columns}×{rows}")
        )
    return gallery


def _update_comparison_layout(columns):
    """Resize comparison cards by changing the number of gallery columns."""
    return gr.Gallery(columns=int(columns), rows=3)


def _features_dataframe(fruits):
    rows = []
    for chili in fruits.items:
        widths = chili.width_fruit_stats
        thickness = chili.mesocarp_thickness_stats
        seeds = chili.seed_stats
        rows.append(
            {
                "ID": chili.id,
                "Name": chili.name,
                "Width BBox": chili.width_bbox,
                "Height BBox": chili.height_bbox,
                "Width Min": widths["min"],
                "Width Max": widths["max"],
                "Average Width": chili.width_fruit,
                "Length": chili.length_fruit,
                "Width (9 segments)": chili.width_fruit_9seg,
                "Length (9 segments)": chili.length_fruit_9seg,
                "Area": chili.area_fruit,
                "Mesocarp Area": chili.mesocarp_area,
                "Fruit Body Area": chili.fruit_body_area,
                "Placenta Area": chili.placenta_area,
                "Min Mesocarp Thickness": thickness["min_mesocarp_thickness"],
                "Average Mesocarp Thickness": thickness["avg_mesocarp_thickness"],
                "Max Mesocarp Thickness": thickness["max_mesocarp_thickness"],
                "Flesh-to-Placenta Ratio": chili.flesh_placenta_ratio,
                "Seed Count": chili.number_seeds,
                "Seed Area Sum": seeds["sum"],
                "Seed Area Min": seeds["min"],
                "Seed Area Max": seeds["max"],
                "Seed Area Average": seeds["avg"],
                "Seed Area Std": seeds["std"],
                "Redness": chili.redness,
                "Color Uniformity": chili.color_uniformity,
                "Mesocarp Wrinkle Index": chili.mesocarp_wrinkle_index,
                "Curvature Index": chili.curvature_index,
                "Seed Density": chili.seed_density,
                "Asymmetry Index": chili.asymmetry_index,
                "Scale Ratio": chili.ratio,
                "Scale Source": chili.scale_source,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def extract_chili_features(image_path):
    """Backward-compatible single-output function used by the original app."""
    if image_path is None or not Path(image_path).exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    fruits = Fruits(
        str(image_path),
        model_detect=model_detect_onnx,
        model_segment=model_segment,
    )
    return _features_dataframe(fruits)


def run_pipeline(
    upload_path,
    sample_name,
    detector_backend,
    chili_method,
    chili_threshold,
    mesocarp_wrinkle_segment_threshold_mm=5.0,
):
    image_path, source = _resolve_input_path(upload_path, sample_name)
    image_rgb = _read_rgb(image_path)
    height, width = image_rgb.shape[:2]

    if height <= DETECTION_CROP_TOP:
        raise gr.Error(
            f"Ảnh chỉ cao {height}px, nhưng detector hiện bỏ {DETECTION_CROP_TOP}px "
            "ở phía trên. Hãy dùng ảnh đúng bố cục dataset hoặc điều chỉnh DetectModel."
        )

    selected_detector = (
        model_detect_onnx
        if detector_backend == "yolov7_onnx"
        else model_detect_threshold
    )

    started_at = time.perf_counter()
    fruits = Fruits(
        str(image_path),
        model_detect=selected_detector,
        model_segment=model_segment,
        chili_method=chili_method,
        chili_threshold=chili_threshold,
        mesocarp_wrinkle_segment_threshold_mm=(
            mesocarp_wrinkle_segment_threshold_mm
        ),
    )
    elapsed = time.perf_counter() - started_at

    detection_roi = image_rgb[DETECTION_CROP_TOP:].copy()
    threshold_mask = get_detection_mask(
        detection_roi,
        threshold=22,
        use_closing=True,
        use_contour=False,
    )
    roi_guide = _draw_roi_guide(image_rgb)
    detection_image = _draw_detection_boxes(image_rgb, fruits)
    crop_gallery = _build_crop_gallery(fruits, image_rgb)
    segmentation_gallery = _build_segmentation_gallery(fruits)
    class_mask_gallery = _build_class_mask_gallery(fruits)
    full_segmentation = _build_full_segmentation(image_rgb, fruits)
    comparison_gallery = _build_comparison_gallery(fruits, image_rgb)
    features = _features_dataframe(fruits)

    ratio_note = f"{fruits.ratio:.6f} mm/pixel (`{fruits.scale_source}`)"
    chili_method_note = f"`{chili_method}`"
    if chili_method == "threshold":
        chili_method_note += f" (threshold offset={int(chili_threshold)})"
    summary = (
        "### Kết quả chạy pipeline\n\n"
        f"- **Nguồn ảnh:** `{source}` — `{image_path.name}`\n"
        f"- **Kích thước:** {width} × {height} px\n"
        f"- **Số quả phát hiện:** {fruits.num_fruit}\n"
        f"- **Detector:** `{detector_backend}`\n"
        f"- **Detection weight:** `{WEIGHT_MODEL_DETECT.name if detector_backend == 'yolov7_onnx' else 'threshold'}`\n"
        f"- **Detection conf/IoU:** `{DEFAULT_CONFIDENCE:.2f}/{DEFAULT_IOU:.2f}`\n"
        f"- **Tỷ lệ hiệu chỉnh:** {ratio_note}\n"
        f"- **Segmentation weight:** `{WEIGHT_MODEL_SEGMENT.name}`\n"
        f"- **Class chili:** {chili_method_note}\n"
        f"- **Ngưỡng mesocarp wrinkle:** {float(mesocarp_wrinkle_segment_threshold_mm):g} mm\n"
        f"- **Thiết bị:** `{DEVICE}`\n"
        f"- **Thời gian:** {elapsed:.2f} giây"
    )

    if fruits.num_fruit == 0:
        summary += (
            "\n\n> Không phát hiện được quả ớt. Hãy kiểm tra Detection ROI, "
            "backend đã chọn và bước segmentation validation."
        )
    if fruits.scale_warning:
        summary += f"\n\n> ⚠️ {fruits.scale_warning}"

    return (
        summary,
        image_rgb,
        roi_guide,
        detection_roi,
        threshold_mask,
        detection_image,
        crop_gallery,
        segmentation_gallery,
        class_mask_gallery,
        full_segmentation,
        comparison_gallery,
        features,
    )


def build_demo():
    with gr.Blocks(title="Chili Pipeline Inspector") as demo:
        gr.Markdown(
            "# 🌶️ Chili Pipeline Inspector\n"
            "Chọn ảnh mẫu trong `data/` hoặc upload ảnh mới. Nếu cả hai đều có, "
            "ảnh upload được ưu tiên. Các tab bên dưới hiển thị tuần tự từng bước "
            "detection, crop, segmentation, class mask và feature extraction."
        )

        with gr.Row():
            sample_input = gr.Dropdown(
                choices=SAMPLE_IMAGES,
                value=SAMPLE_IMAGES[0] if SAMPLE_IMAGES else None,
                label="Ảnh mẫu từ data/",
            )
            upload_input = gr.Image(
                type="filepath",
                label="Hoặc upload ảnh trực tiếp",
            )

        with gr.Row():
            detector_backend_input = gr.Radio(
                choices=["yolov7_onnx", "threshold"],
                value="yolov7_onnx",
                label="Object detector",
                info="YOLOv7 ONNX chạy CPU; threshold là fallback.",
            )
            chili_method_input = gr.Radio(
                choices=["threshold", "yolo"],
                value="threshold",
                label="Phương pháp tạo class chili",
                info="Các class seed/outer/PCP/endocarp vẫn dùng YOLO.",
            )
            chili_threshold_input = gr.Slider(
                minimum=0,
                maximum=100,
                value=16,
                step=1,
                label="Threshold offset cho class chili",
            )
            mesocarp_wrinkle_threshold_input = gr.Slider(
                minimum=0.5,
                maximum=20.0,
                value=5.0,
                step=0.5,
                label="Mesocarp wrinkle segment threshold (mm)",
                info="Tham số hình học riêng, không liên quan threshold độ đỏ.",
            )

        run_button = gr.Button("Chạy toàn bộ pipeline", variant="primary")
        summary_output = gr.Markdown()

        with gr.Tab("1. Input & ROI"):
            with gr.Row():
                original_output = gr.Image(
                    label="Ảnh đầu vào",
                    height=460,
                    interactive=False,
                )
                roi_output = gr.Image(
                    label="Detection ROI — vùng tối phía trên bị bỏ qua",
                    height=460,
                    interactive=False,
                )
            detection_roi_output = gr.Image(
                label="Ảnh ROI thực tế được truyền vào detector",
                height=460,
                interactive=False,
            )

        with gr.Tab("2. Detection"):
            with gr.Row():
                threshold_mask_output = gr.Image(
                    label="Red-channel threshold mask tham chiếu",
                    height=460,
                    interactive=False,
                )
                detection_output = gr.Image(
                    label="Bounding box xanh lá; crop padding màu vàng",
                    height=460,
                    interactive=False,
                )

        with gr.Tab("3. Chili Crops"):
            gr.Markdown(
                "**Chili crop là gì?** Detector tìm bounding box của từng quả, "
                "mở rộng thêm 50 px ở mỗi cạnh rồi cắt thành ảnh riêng. Ảnh crop "
                "sau đó được xoay để cuống và thân quả có hướng nhất quán trước "
                "khi chạy YOLO segmentation. Mỗi quả bên dưới có ảnh trước và sau xoay."
            )
            crop_output = gr.Gallery(
                label="Raw padded crop và crop sau chuẩn hóa hướng",
                columns=4,
                rows=2,
                height=520,
                interactive=False,
            )

        with gr.Tab("4. Segmentation"):
            gr.Markdown(
                "**Chú thích màu segmentation:** mỗi màu phủ tương ứng với một class."
            )
            gr.HTML(
                _color_legend_html(
                    [
                        ("seed", CLASS_COLORS[0]),
                        ("outer", CLASS_COLORS[1]),
                        ("PCP", CLASS_COLORS[2]),
                        ("endocarp", CLASS_COLORS[3]),
                        ("chili", CLASS_COLORS[4]),
                    ]
                )
            )
            segmentation_output = gr.Gallery(
                label="Hybrid segmentation overlay",
                columns=4,
                rows=2,
                height=520,
                interactive=False,
            )

        with gr.Tab("5. Class Masks"):
            gr.Markdown(
                "Mỗi item bên dưới tương ứng với **một quả ớt**. Ảnh đầu tiên là "
                "`original`, sau đó lần lượt là overlay của từng class. Quả dọc, "
                "có tile đủ hẹp, sẽ xếp thành một hàng 1×8; nếu tổng chiều rộng "
                "không đủ thì tự chuyển sang lưới 4×2."
            )
            gr.HTML(
                _color_legend_html(
                    [
                        ("seed", CLASS_COLORS[0]),
                        ("outer", CLASS_COLORS[1]),
                        ("PCP", CLASS_COLORS[2]),
                        ("endocarp", CLASS_COLORS[3]),
                        ("chili", CLASS_COLORS[4]),
                        ("pedicel", EXTRA_MASKS["pedicel"]),
                        ("placenta", EXTRA_MASKS["placenta"]),
                    ]
                )
            )
            mask_output = gr.Gallery(
                label="Một contact sheet cho mỗi quả",
                columns=1,
                rows=1,
                height=900,
                interactive=False,
            )

        with gr.Tab("6. Final Result & Comparison"):
            gr.Markdown(
                "Ảnh đầu tiên chiếu toàn bộ mask của từng quả về đúng vị trí trên "
                "ảnh gốc. Các card tiếp theo đặt crop gốc, segmentation và thông số "
                "cạnh nhau để so sánh trực tiếp giữa các quả."
            )
            full_segmentation_output = gr.Image(
                label="Segmentation tổng hợp trên toàn ảnh",
                height=600,
                interactive=False,
            )
            comparison_columns = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Số card mỗi hàng — ít cột hơn = card lớn hơn",
            )
            comparison_output = gr.Gallery(
                label="So sánh trực quan từng quả",
                columns=3,
                rows=3,
                height=820,
                interactive=False,
            )
            with gr.Accordion("Bảng feature chi tiết", open=False):
                feature_output = gr.Dataframe(
                    headers=FEATURE_COLUMNS,
                    label="Dữ liệu số đầy đủ",
                    interactive=False,
                )

        run_button.click(
            fn=run_pipeline,
            inputs=[
                upload_input,
                sample_input,
                detector_backend_input,
                chili_method_input,
                chili_threshold_input,
                mesocarp_wrinkle_threshold_input,
            ],
            outputs=[
                summary_output,
                original_output,
                roi_output,
                detection_roi_output,
                threshold_mask_output,
                detection_output,
                crop_output,
                segmentation_output,
                mask_output,
                full_segmentation_output,
                comparison_output,
                feature_output,
            ],
        )

        comparison_columns.change(
            fn=_update_comparison_layout,
            inputs=comparison_columns,
            outputs=comparison_output,
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
