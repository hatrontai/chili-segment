import os
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch

from src.detect import DetectModel, get_mask as get_detection_mask
from src.object import CLASS_COLORS, CLASS_NAMES, Fruits, draw_full_mask_with_labels
from src.segment import SegmentModel


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DETECTION_CROP_TOP = 540
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WEIGHT_MODEL_SEGMENT = Path(
    os.getenv(
        "CHILI_SEGMENT_WEIGHT",
        str(BASE_DIR / "pretrain" / "yolov8_segment_v2.pt"),
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
    "Flesh Area",
    "Fruit Body Area",
    "Placenta Area",
    "Max Thickness",
    "Average Thickness",
    "Flesh/Placenta Ratio",
    "Seed Count",
    "Seed Area Sum",
    "Seed Area Min",
    "Seed Area Max",
    "Seed Area Average",
    "Seed Area Std",
    "Redness",
    "Color Uniformity",
    "Wrinkle",
    "Curvature Index",
    "Seed Density",
    "Asymmetry Index",
    "Scale Ratio",
]

EXTRA_MASKS = {
    "stem": (255, 165, 0),
    "true_placenta": (0, 215, 255),
}


if not WEIGHT_MODEL_SEGMENT.exists():
    raise FileNotFoundError(
        f"Segmentation weight not found: {WEIGHT_MODEL_SEGMENT}"
    )

print(f"Loading pipeline models on device {DEVICE}...")
model_detect = DetectModel(model_name="none")
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
    """Overlay one binary mask on a crop without modifying its geometry."""
    output = image_rgb.copy()
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return output

    colored = output.copy()
    colored[mask_bool] = color
    output = cv2.addWeighted(colored, 0.55, output, 0.45, 0)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, color, 2)
    return output


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
        ("placenta", "mask_placenta", CLASS_COLORS[2]),
        ("endocarp", "mask_endocarp", CLASS_COLORS[3]),
        ("chili", "mask_chili", CLASS_COLORS[4]),
        ("stem", "mask_stem", EXTRA_MASKS["stem"]),
        ("true placenta", "mask_true_placenta", EXTRA_MASKS["true_placenta"]),
    ]

    for chili in fruits.items:
        for label, attribute, color in mask_attributes:
            mask = getattr(chili, attribute, None)
            if mask is None or not np.any(mask > 0):
                continue
            rendered = _overlay_mask(chili.img, mask, color)
            area_px = int(np.count_nonzero(mask))
            gallery.append(
                (rendered, f"Chili {chili.id} · {label} · {area_px:,} px")
            )
    return gallery


def _features_dataframe(fruits):
    rows = []
    for chili in fruits.items:
        widths = chili.width_fruit_stats
        thickness = chili.thickness_stats
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
                "Flesh Area": chili.endocarp_area,
                "Fruit Body Area": chili.fruit_body_area,
                "Placenta Area": chili.placenta_area,
                "Max Thickness": thickness["max_thickness"],
                "Average Thickness": thickness["avg_thickness"],
                "Flesh/Placenta Ratio": chili.flesh_placenta_ratio,
                "Seed Count": chili.number_seeds,
                "Seed Area Sum": seeds["sum"],
                "Seed Area Min": seeds["min"],
                "Seed Area Max": seeds["max"],
                "Seed Area Average": seeds["avg"],
                "Seed Area Std": seeds["std"],
                "Redness": chili.redness,
                "Color Uniformity": chili.color_uniformity,
                "Wrinkle": chili.wrinkle,
                "Curvature Index": chili.curvature_index,
                "Seed Density": chili.seed_density,
                "Asymmetry Index": chili.asymmetry_index,
                "Scale Ratio": chili.ratio,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def extract_chili_features(image_path):
    """Backward-compatible single-output function used by the original app."""
    if image_path is None or not Path(image_path).exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    fruits = Fruits(
        str(image_path),
        model_detect=model_detect,
        model_segment=model_segment,
    )
    return _features_dataframe(fruits)


def run_pipeline(upload_path, sample_name):
    image_path, source = _resolve_input_path(upload_path, sample_name)
    image_rgb = _read_rgb(image_path)
    height, width = image_rgb.shape[:2]

    if height <= DETECTION_CROP_TOP:
        raise gr.Error(
            f"Ảnh chỉ cao {height}px, nhưng detector hiện bỏ {DETECTION_CROP_TOP}px "
            "ở phía trên. Hãy dùng ảnh đúng bố cục dataset hoặc điều chỉnh DetectModel."
        )

    started_at = time.perf_counter()
    fruits = Fruits(
        str(image_path),
        model_detect=model_detect,
        model_segment=model_segment,
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
    features = _features_dataframe(fruits)

    ratio_note = (
        "1.0 (fallback; thường là không đọc được QR)"
        if fruits.ratio == 1.0
        else f"{fruits.ratio:.6f} mm/pixel"
    )
    summary = (
        "### Kết quả chạy pipeline\n\n"
        f"- **Nguồn ảnh:** `{source}` — `{image_path.name}`\n"
        f"- **Kích thước:** {width} × {height} px\n"
        f"- **Số quả phát hiện:** {fruits.num_fruit}\n"
        f"- **Tỷ lệ hiệu chỉnh:** {ratio_note}\n"
        f"- **Segmentation weight:** `{WEIGHT_MODEL_SEGMENT.name}`\n"
        f"- **Thiết bị:** `{DEVICE}`\n"
        f"- **Thời gian:** {elapsed:.2f} giây"
    )

    if fruits.num_fruit == 0:
        summary += (
            "\n\n> Không phát hiện được quả ớt. Hãy kiểm tra tab Detection ROI "
            "để xác nhận ảnh có đúng bố cục mà threshold detector yêu cầu hay không."
        )

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

        run_button = gr.Button("Chạy toàn bộ pipeline", variant="primary")
        summary_output = gr.Markdown()

        with gr.Tab("1. Input & ROI"):
            with gr.Row():
                original_output = gr.Image(label="Ảnh đầu vào", interactive=False)
                roi_output = gr.Image(
                    label="Detection ROI — vùng tối phía trên bị bỏ qua",
                    interactive=False,
                )
            detection_roi_output = gr.Image(
                label="Ảnh thực tế được truyền vào threshold detector",
                interactive=False,
            )

        with gr.Tab("2. Detection"):
            with gr.Row():
                threshold_mask_output = gr.Image(
                    label="Red-channel threshold mask sau morphology closing",
                    interactive=False,
                )
                detection_output = gr.Image(
                    label="Bounding box xanh lá; crop padding màu vàng",
                    interactive=False,
                )

        with gr.Tab("3. Chili Crops"):
            crop_output = gr.Gallery(
                label="Các crop đã chuẩn hóa hướng",
                columns=3,
            )

        with gr.Tab("4. Segmentation"):
            segmentation_output = gr.Gallery(
                label="YOLO segmentation overlay",
                columns=3,
            )

        with gr.Tab("5. Class Masks"):
            mask_output = gr.Gallery(
                label="Mask riêng của từng class và stem hậu xử lý",
                columns=3,
            )

        with gr.Tab("6. Features"):
            feature_output = gr.Dataframe(
                headers=FEATURE_COLUMNS,
                label="Đặc trưng phenotyping",
                interactive=False,
            )

        run_button.click(
            fn=run_pipeline,
            inputs=[upload_input, sample_input],
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
                feature_output,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
