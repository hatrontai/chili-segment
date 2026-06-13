from pathlib import Path

import cv2
import numpy as np


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
    "debug_dir",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    column
    for class_name in CLASS_NAMES.values()
    for column in (
        f"{class_name}_mask_area_px",
        f"{class_name}_mask_ratio",
    )
]


def _write_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _align_mask(mask, target_shape):
    mask = np.asarray(mask, dtype=np.uint8)
    target_height, target_width = target_shape

    if mask.shape == target_shape:
        return mask

    # Chili.get_width_height_bbox may rotate the crop clockwise.
    rotated = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    if rotated.shape == target_shape:
        return rotated

    return cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )


def collect_class_masks(chili):
    """Merge all segmentation instances into one binary mask per class."""
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


def build_mask_overlay(image, class_masks, alpha=0.45):
    """Overlay all non-empty class masks using stable debug colors."""
    color_layer = image.copy()
    covered = np.zeros(image.shape[:2], dtype=bool)

    # Draw large enclosing classes first so small regions remain visible.
    for class_id in sorted(class_masks, reverse=True):
        mask = class_masks[class_id]
        selected = mask > 0
        color_layer[selected] = CLASS_COLORS[class_id]
        covered |= selected

    output = image.copy()
    blended = cv2.addWeighted(image, 1.0 - alpha, color_layer, alpha, 0)
    output[covered] = blended[covered]
    return output


def skeletonize_mask(mask):
    """Create a one-pixel morphological skeleton without extra packages."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(binary) > 0:
        eroded = cv2.erode(binary, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(
            skeleton,
            cv2.subtract(binary, opened),
        )
        binary = eroded

    return skeleton


def build_skeleton_overlay(image, mask):
    skeleton = skeletonize_mask(mask)
    output = image.copy()
    output[skeleton > 0] = (0, 255, 255)
    return skeleton, output


def feature_row(chili, source_image, class_masks, output_dir):
    fruit_pixels = int(np.count_nonzero(chili.mask))
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
        "debug_dir": str(output_dir),
    }

    denominator = fruit_pixels if fruit_pixels > 0 else 1
    for class_id, class_name in CLASS_NAMES.items():
        area = int(np.count_nonzero(class_masks[class_id]))
        row[f"{class_name}_mask_area_px"] = area
        row[f"{class_name}_mask_ratio"] = round(area / denominator, 6)

    return row


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


def build_debug_summary(chili, overlay, skeleton_overlay):
    panel_size = (520, 360)
    fruit_mask_rgb = cv2.cvtColor(chili.mask, cv2.COLOR_GRAY2RGB)
    panels = [
        _label_panel(_fit_panel(chili.img, panel_size), "Crop"),
        _label_panel(_fit_panel(overlay, panel_size), "Segmentation overlay"),
        _label_panel(_fit_panel(fruit_mask_rgb, panel_size), "Feature mask"),
        _label_panel(_fit_panel(skeleton_overlay, panel_size), "Skeleton overlay"),
    ]
    image_grid = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))

    text_canvas = np.full((190, image_grid.shape[1], 3), 255, dtype=np.uint8)
    lines = [
        f"id={chili.id} name={chili.name}",
        f"bbox={chili.bbox}",
        f"bbox_size=({chili.width_bbox}, {chili.height_bbox})",
        f"width_mean={chili.width_fruit} length={chili.length_fruit}",
        f"area={chili.area_fruit} redness={chili.redness}",
        f"wrinkle={chili.wrinkle} seeds={chili.number_seeds}",
        f"scale_ratio={chili.ratio}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            text_canvas,
            line,
            (15, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )

    return np.vstack((image_grid, text_canvas))


def save_chili_debug(chili, source_image, output_dir):
    """Save each intermediate artifact for one detected chili."""
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    class_masks = collect_class_masks(chili)
    overlay = build_mask_overlay(chili.img, class_masks)
    skeleton, skeleton_overlay = build_skeleton_overlay(chili.img, chili.mask)
    summary = build_debug_summary(chili, overlay, skeleton_overlay)

    _write_rgb(output_dir / "crop.jpg", chili.img)
    for class_id, class_name in CLASS_NAMES.items():
        cv2.imwrite(
            str(mask_dir / f"{class_id:02d}_{class_name}.png"),
            class_masks[class_id],
        )
    cv2.imwrite(str(output_dir / "feature_mask.png"), chili.mask)
    cv2.imwrite(str(output_dir / "skeleton.png"), skeleton)
    _write_rgb(output_dir / "mask_overlay.jpg", overlay)
    _write_rgb(output_dir / "skeleton_overlay.jpg", skeleton_overlay)
    _write_rgb(output_dir / "debug_summary.jpg", summary)

    return feature_row(chili, source_image, class_masks, output_dir)
