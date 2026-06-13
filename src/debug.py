from pathlib import Path

import cv2
import numpy as np

from src.measurement import clean_mask, color_metrics, length_metrics


CLASS_NAMES = {
    0: "seed",
    1: "outer",
    2: "placenta",
    3: "endocarp",
    4: "chili",
}

CLASS_COLORS = {
    0: (220, 60, 60),
    1: (50, 180, 70),
    2: (70, 90, 220),
    3: (230, 190, 40),
    4: (190, 70, 190),
}

BASE_FEATURE_COLUMNS = [
    "source_image",
    "chili_id",
    "name_image",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "width_bbox",
    "height_bbox",
    "average_width",
    "length",
    "area",
    "redness",
    "wrinkle",
    "number_seeds",
    "scale_ratio",
    "fruit_mask_area_px",
    "clean_fruit_mask_area_px",
    "pixel_per_cm",
    "calibration_source",
    "skeleton_length_px",
    "bbox_length_px",
    "contour_perimeter_px",
    "mask_bbox_width_px",
    "mask_bbox_height_px",
    "skeleton_length_cm",
    "bbox_length_cm",
    "contour_perimeter_cm",
    "fruit_area_cm2",
    "debug_dir",
]

MASK_FEATURE_COLUMNS = [
    column
    for class_name in CLASS_NAMES.values()
    for column in (
        f"{class_name}_mask_area_raw_px",
        f"{class_name}_mask_area_clean_px",
        f"{class_name}_mask_ratio",
    )
]

COLOR_FEATURE_COLUMNS = [
    "color_valid_pixels",
    "shadow_threshold_v",
    "red_pixel_count",
    "red_ratio_robust",
] + [
    f"{color_name}_{channel}_{statistic}"
    for color_name, channels in (
        ("hsv", ("h", "s", "v")),
        ("lab", ("l", "a", "b")),
    )
    for channel in channels
    for statistic in ("mean", "std")
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + MASK_FEATURE_COLUMNS + COLOR_FEATURE_COLUMNS


def _write_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _write_metrics(path, values):
    lines = [f"{key}: {value}" for key, value in values.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _align_mask(mask, target_shape):
    mask = np.asarray(mask, dtype=np.uint8)
    target_height, target_width = target_shape
    if mask.shape == target_shape:
        return mask

    # Chili may rotate its crop after segmentation.
    rotated = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    if rotated.shape == target_shape:
        return rotated

    return cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )


def collect_class_masks(chili):
    """Merge segmentation instances into one binary mask per class."""
    image_shape = chili.img.shape[:2]
    class_masks = {
        class_id: np.zeros(image_shape, dtype=np.uint8)
        for class_id in CLASS_NAMES
    }

    for item in chili.rseg.get("masks_data", []):
        class_id = int(item["class_id"])
        if class_id not in class_masks:
            continue
        mask = _align_mask(item["mask"], image_shape)
        class_masks[class_id] = cv2.bitwise_or(
            class_masks[class_id],
            np.where(mask > 0, 255, 0).astype(np.uint8),
        )
    return class_masks


def clean_class_masks(class_masks, min_component_area=100):
    cleaned = {}
    for class_id, mask in class_masks.items():
        class_min_area = (
            max(4, min_component_area // 4)
            if class_id == 0 else min_component_area
        )
        cleaned[class_id] = clean_mask(
            mask,
            min_area=class_min_area,
            keep_largest=(class_id == 4),
        )
    return cleaned


def build_mask_overlay(image, class_masks, alpha=0.45):
    color_layer = image.copy()
    covered = np.zeros(image.shape[:2], dtype=bool)

    # Draw large enclosing classes first, then small internal regions.
    for class_id in sorted(class_masks, reverse=True):
        selected = class_masks[class_id] > 0
        color_layer[selected] = CLASS_COLORS[class_id]
        covered |= selected

    output = image.copy()
    blended = cv2.addWeighted(image, 1.0 - alpha, color_layer, alpha, 0)
    output[covered] = blended[covered]
    return output


def build_skeleton_overlay(image, skeleton, path=None):
    output = image.copy()
    output[skeleton > 0] = (0, 255, 255)

    if path:
        points = np.array(
            [(x, y) for y, x in path],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(output, [points], False, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(output, tuple(points[0, 0]), 4, (0, 255, 0), -1)
        cv2.circle(output, tuple(points[-1, 0]), 4, (255, 0, 255), -1)
    return output


def build_color_overlay(image, valid_mask, red_mask):
    output = image.copy()
    invalid = valid_mask == 0
    output[invalid] = (output[invalid] * 0.35).astype(np.uint8)
    valid = valid_mask > 0
    output[valid] = (
        0.65 * output[valid] + 0.35 * np.array([0, 180, 255])
    ).astype(np.uint8)
    output[red_mask > 0] = (255, 0, 0)
    return output


def _fit_panel(image, size):
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(image.shape[1] * scale)),
            max(1, int(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _label_panel(image, label):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(
        output,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def build_before_after(before, after):
    panel_size = (520, 360)
    before_rgb = cv2.cvtColor(before, cv2.COLOR_GRAY2RGB)
    after_rgb = cv2.cvtColor(after, cv2.COLOR_GRAY2RGB)
    return np.hstack((
        _label_panel(_fit_panel(before_rgb, panel_size), "Before"),
        _label_panel(_fit_panel(after_rgb, panel_size), "After"),
    ))


def build_debug_summary(
    chili,
    clean_overlay,
    before_after,
    skeleton_overlay,
    color_overlay,
    lengths,
    colors,
    pixel_per_cm,
):
    panel_size = (520, 360)
    panels = [
        _label_panel(_fit_panel(chili.img, panel_size), "Crop"),
        _label_panel(_fit_panel(clean_overlay, panel_size), "Clean mask overlay"),
        _label_panel(_fit_panel(before_after, panel_size), "Fruit mask before / after"),
        _label_panel(_fit_panel(skeleton_overlay, panel_size), "Longest skeleton path"),
        _label_panel(_fit_panel(color_overlay, panel_size), "Valid color / red pixels"),
        _label_panel(_fit_panel(chili.img, panel_size), "Feature summary"),
    ]
    image_grid = np.vstack((
        np.hstack(panels[:2]),
        np.hstack(panels[2:4]),
        np.hstack(panels[4:]),
    ))

    text_canvas = np.full((240, image_grid.shape[1], 3), 255, dtype=np.uint8)
    calibrated_length = (
        f"{lengths['skeleton_length_px'] / pixel_per_cm:.3f} cm"
        if pixel_per_cm else "N/A"
    )
    lines = [
        f"id={chili.id} name={chili.name}",
        f"bbox_xywh={chili.bbox}",
        f"bbox_size_legacy=({chili.width_bbox}, {chili.height_bbox})",
        f"legacy_width={chili.width_fruit} legacy_length={chili.length_fruit}",
        f"legacy_area={chili.area_fruit} legacy_redness={chili.redness}",
        f"skeleton={lengths['skeleton_length_px']:.2f}px ({calibrated_length})",
        f"bbox_length={lengths['bbox_length_px']:.2f}px contour={lengths['contour_perimeter_px']:.2f}px",
        f"red_ratio={colors['red_ratio_robust']:.4f} valid_color_px={colors['color_valid_pixels']}",
        f"pixel_per_cm={pixel_per_cm if pixel_per_cm else 'N/A'}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            text_canvas,
            line,
            (15, 26 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((image_grid, text_canvas))


def save_detection_overview(source_image, fruits, output_path):
    image = cv2.imread(str(source_image))
    if image is None:
        raise ValueError(f"Cannot read image: {source_image}")

    for chili in fruits.items:
        x, y, width, height = map(int, chili.bbox)
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 4)
        cv2.putText(
            image,
            f"chili_{chili.id:03d}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def feature_row(
    chili,
    source_image,
    raw_class_masks,
    cleaned_class_masks,
    clean_fruit_mask,
    lengths,
    colors,
    output_dir,
    pixel_per_cm=None,
    calibration_source="none",
):
    fruit_pixels = int(np.count_nonzero(chili.mask))
    clean_fruit_pixels = int(np.count_nonzero(clean_fruit_mask))

    def to_cm(value):
        return round(value / pixel_per_cm, 4) if pixel_per_cm else ""

    row = {
        "source_image": str(source_image),
        "chili_id": chili.id,
        "name_image": chili.name,
        "bbox_x": chili.bbox[0],
        "bbox_y": chili.bbox[1],
        "bbox_w": chili.bbox[2],
        "bbox_h": chili.bbox[3],
        "width_bbox": chili.width_bbox,
        "height_bbox": chili.height_bbox,
        "average_width": chili.width_fruit,
        "length": chili.length_fruit,
        "area": chili.area_fruit,
        "redness": chili.redness,
        "wrinkle": chili.wrinkle,
        "number_seeds": chili.number_seeds,
        "scale_ratio": chili.ratio,
        "fruit_mask_area_px": fruit_pixels,
        "clean_fruit_mask_area_px": clean_fruit_pixels,
        "pixel_per_cm": pixel_per_cm if pixel_per_cm else "",
        "calibration_source": calibration_source,
        "skeleton_length_px": round(lengths["skeleton_length_px"], 4),
        "bbox_length_px": round(lengths["bbox_length_px"], 4),
        "contour_perimeter_px": round(lengths["contour_perimeter_px"], 4),
        "mask_bbox_width_px": round(lengths["mask_bbox_width_px"], 4),
        "mask_bbox_height_px": round(lengths["mask_bbox_height_px"], 4),
        "skeleton_length_cm": to_cm(lengths["skeleton_length_px"]),
        "bbox_length_cm": to_cm(lengths["bbox_length_px"]),
        "contour_perimeter_cm": to_cm(lengths["contour_perimeter_px"]),
        "fruit_area_cm2": (
            round(clean_fruit_pixels / (pixel_per_cm ** 2), 4)
            if pixel_per_cm else ""
        ),
        "debug_dir": str(output_dir),
    }

    denominator = max(1, clean_fruit_pixels)
    for class_id, class_name in CLASS_NAMES.items():
        raw_area = int(np.count_nonzero(raw_class_masks[class_id]))
        clean_area = int(np.count_nonzero(cleaned_class_masks[class_id]))
        row[f"{class_name}_mask_area_raw_px"] = raw_area
        row[f"{class_name}_mask_area_clean_px"] = clean_area
        row[f"{class_name}_mask_ratio"] = round(clean_area / denominator, 6)

    for key in COLOR_FEATURE_COLUMNS:
        row[key] = colors[key]
    return row


def save_chili_debug(
    chili,
    source_image,
    output_dir,
    pixel_per_cm=None,
    calibration_source="none",
    min_component_area=100,
    min_branch_length=12.0,
):
    """Save inspectable outputs for all five evaluation steps."""
    output_dir = Path(output_dir)
    step1_dir = output_dir / "step1_debug"
    step2_dir = output_dir / "step2_masks"
    step3_dir = output_dir / "step3_length"
    step4_dir = output_dir / "step4_color"
    step5_dir = output_dir / "step5_measurement"
    for path in (step1_dir, step2_dir, step3_dir, step4_dir, step5_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw_class_masks = collect_class_masks(chili)
    cleaned_class_masks = clean_class_masks(
        raw_class_masks,
        min_component_area,
    )
    raw_fruit_mask = (
        raw_class_masks[4]
        if np.count_nonzero(raw_class_masks[4]) > 0
        else chili.mask
    )
    clean_fruit_mask = clean_mask(
        raw_fruit_mask,
        min_area=min_component_area,
        keep_largest=True,
    )

    raw_overlay = build_mask_overlay(chili.img, raw_class_masks)
    clean_overlay = build_mask_overlay(chili.img, cleaned_class_masks)
    before_after = build_before_after(raw_fruit_mask, clean_fruit_mask)
    lengths = length_metrics(clean_fruit_mask, min_branch_length)
    skeleton_overlay = build_skeleton_overlay(
        chili.img,
        lengths["pruned_skeleton"],
        lengths["path"],
    )
    colors = color_metrics(chili.img, clean_fruit_mask)
    color_overlay = build_color_overlay(
        chili.img,
        colors["valid_mask"],
        colors["red_mask"],
    )
    summary = build_debug_summary(
        chili,
        clean_overlay,
        before_after,
        skeleton_overlay,
        color_overlay,
        lengths,
        colors,
        pixel_per_cm,
    )

    _write_rgb(step1_dir / "crop.jpg", chili.img)
    _write_rgb(step1_dir / "raw_mask_overlay.jpg", raw_overlay)

    for class_id, class_name in CLASS_NAMES.items():
        cv2.imwrite(
            str(step2_dir / f"{class_id:02d}_{class_name}_before.png"),
            raw_class_masks[class_id],
        )
        cv2.imwrite(
            str(step2_dir / f"{class_id:02d}_{class_name}_after.png"),
            cleaned_class_masks[class_id],
        )
    cv2.imwrite(str(step2_dir / "fruit_mask_before.png"), raw_fruit_mask)
    cv2.imwrite(str(step2_dir / "fruit_mask_after.png"), clean_fruit_mask)
    _write_rgb(step2_dir / "clean_mask_overlay.jpg", clean_overlay)
    _write_rgb(step2_dir / "fruit_mask_before_after.jpg", before_after)

    cv2.imwrite(str(step3_dir / "skeleton_raw.png"), lengths["raw_skeleton"])
    cv2.imwrite(str(step3_dir / "skeleton_pruned.png"), lengths["pruned_skeleton"])
    _write_rgb(step3_dir / "skeleton_path_overlay.jpg", skeleton_overlay)
    _write_metrics(
        step3_dir / "length_comparison.txt",
        {
            key: round(lengths[key], 4)
            for key in (
                "skeleton_length_px",
                "bbox_length_px",
                "contour_perimeter_px",
            )
        },
    )

    cv2.imwrite(str(step4_dir / "color_valid_mask.png"), colors["valid_mask"])
    cv2.imwrite(str(step4_dir / "red_pixels_mask.png"), colors["red_mask"])
    _write_rgb(step4_dir / "color_overlay.jpg", color_overlay)
    _write_metrics(
        step4_dir / "color_features.txt",
        {
            key: value
            for key, value in colors.items()
            if key not in {"valid_mask", "red_mask"}
        },
    )

    measurement_values = {
        "pixel_per_cm": pixel_per_cm if pixel_per_cm else "N/A",
        "calibration_source": calibration_source,
        "length_cm": (
            round(lengths["skeleton_length_px"] / pixel_per_cm, 4)
            if pixel_per_cm else "N/A"
        ),
        "area_cm2": (
            round(np.count_nonzero(clean_fruit_mask) / (pixel_per_cm ** 2), 4)
            if pixel_per_cm else "N/A"
        ),
    }
    _write_metrics(step5_dir / "calibrated_measurements.txt", measurement_values)
    _write_rgb(output_dir / "debug_summary.jpg", summary)

    print(f"    STEP 1: {step1_dir}")
    print(f"    STEP 2: {step2_dir}")
    print(f"    STEP 3: {step3_dir}")
    print(f"    STEP 4: {step4_dir}")
    print(f"    STEP 5: {step5_dir}")
    print(f"    SUMMARY: {output_dir / 'debug_summary.jpg'}")

    return feature_row(
        chili,
        source_image,
        raw_class_masks,
        cleaned_class_masks,
        clean_fruit_mask,
        lengths,
        colors,
        output_dir,
        pixel_per_cm,
        calibration_source,
    )
