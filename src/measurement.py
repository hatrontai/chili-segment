import heapq
import math

import cv2
import numpy as np


NEIGHBORS = [
    (-1, -1, math.sqrt(2)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2)),
]


def binary_mask(mask):
    return np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)


def fill_mask_holes(mask):
    """Fill enclosed background regions without SciPy."""
    binary = binary_mask(mask)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(binary, holes)


def remove_small_components(mask, min_area=100, keep_largest=False):
    binary = binary_mask(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros_like(binary)

    components = [
        (label, int(stats[label, cv2.CC_STAT_AREA]))
        for label in range(1, count)
    ]
    if keep_largest and components:
        components = [max(components, key=lambda item: item[1])]

    for label, area in components:
        if area >= min_area:
            output[labels == label] = 255
    return output


def clean_mask(
    mask,
    open_kernel=3,
    close_kernel=7,
    min_area=100,
    keep_largest=False,
):
    """Open, close, fill holes, then remove small connected components."""
    result = binary_mask(mask)

    if open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel, open_kernel),
        )
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    if close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel, close_kernel),
        )
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    result = fill_mask_holes(result)
    return remove_small_components(result, min_area, keep_largest)


def skeletonize_mask(mask):
    """Create a connected one-pixel skeleton with Zhang-Suen thinning."""
    image = (binary_mask(mask) > 0).astype(np.uint8)

    while True:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                preserve_a = p2 * p4 * p6
                preserve_b = p4 * p6 * p8
            else:
                preserve_a = p2 * p4 * p8
                preserve_b = p2 * p6 * p8

            remove = (
                (image == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & (preserve_a == 0)
                & (preserve_b == 0)
            )
            if np.any(remove):
                image[remove] = 0
                changed = True

        if not changed:
            break

    return (image * 255).astype(np.uint8)


def _pixel_neighbors(point, pixels):
    y, x = point
    for dy, dx, weight in NEIGHBORS:
        neighbor = (y + dy, x + dx)
        if neighbor in pixels:
            yield neighbor, weight


def prune_skeleton(skeleton, min_branch_length=12.0, max_rounds=20):
    """Remove short endpoint-to-junction branches while preserving trunks."""
    pruned = binary_mask(skeleton)

    for _ in range(max_rounds):
        pixels = set(map(tuple, np.argwhere(pruned > 0)))
        if not pixels:
            break
        degrees = {
            point: sum(1 for _ in _pixel_neighbors(point, pixels))
            for point in pixels
        }
        endpoints = [point for point, degree in degrees.items() if degree == 1]
        remove = set()

        for endpoint in endpoints:
            path = [endpoint]
            previous = None
            current = endpoint
            length = 0.0

            while True:
                next_nodes = [
                    item
                    for item in _pixel_neighbors(current, pixels)
                    if item[0] != previous
                ]
                if len(next_nodes) != 1:
                    break
                next_point, weight = next_nodes[0]
                previous, current = current, next_point
                path.append(current)
                length += weight
                if degrees.get(current, 0) != 2:
                    break

            if degrees.get(current, 0) >= 3 and length < min_branch_length:
                remove.update(path[:-1])

        if not remove:
            break
        for y, x in remove:
            pruned[y, x] = 0

    return pruned


def _dijkstra(pixels, start):
    distances = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]

    while queue:
        distance, point = heapq.heappop(queue)
        if distance != distances.get(point):
            continue
        for neighbor, weight in _pixel_neighbors(point, pixels):
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = point
                heapq.heappush(queue, (candidate, neighbor))
    return distances, previous


def longest_skeleton_path(skeleton):
    """Estimate graph diameter with two weighted Dijkstra passes."""
    binary = remove_small_components(skeleton, min_area=1, keep_largest=True)
    pixels = set(map(tuple, np.argwhere(binary > 0)))
    if not pixels:
        return 0.0, []

    start = next(iter(pixels))
    distances, _ = _dijkstra(pixels, start)
    endpoint_a = max(distances, key=distances.get)
    distances, previous = _dijkstra(pixels, endpoint_a)
    endpoint_b = max(distances, key=distances.get)

    path = [endpoint_b]
    while path[-1] != endpoint_a:
        parent = previous.get(path[-1])
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    return float(distances[endpoint_b]), path


def length_metrics(mask, min_branch_length=12.0):
    clean = binary_mask(mask)
    raw_skeleton = skeletonize_mask(clean)
    pruned_skeleton = prune_skeleton(raw_skeleton, min_branch_length)
    skeleton_length, path = longest_skeleton_path(pruned_skeleton)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_perimeter = (
        float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
        if contours else 0.0
    )
    points = cv2.findNonZero(clean)
    if points is None:
        bbox_width = bbox_height = bbox_length = 0.0
    else:
        _, _, bbox_width, bbox_height = cv2.boundingRect(points)
        bbox_length = float(max(bbox_width, bbox_height))

    return {
        "raw_skeleton": raw_skeleton,
        "pruned_skeleton": pruned_skeleton,
        "path": path,
        "skeleton_length_px": skeleton_length,
        "bbox_length_px": bbox_length,
        "mask_bbox_width_px": float(bbox_width),
        "mask_bbox_height_px": float(bbox_height),
        "contour_perimeter_px": contour_perimeter,
    }


def color_metrics(image_rgb, mask):
    """Measure color inside the mask after excluding strong shadows."""
    selected = binary_mask(mask) > 0
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    values = hsv[:, :, 2][selected]
    if values.size == 0:
        valid = np.zeros(selected.shape, dtype=bool)
        shadow_threshold = 0
    else:
        shadow_threshold = max(30, int(np.percentile(values, 10)))
        valid = selected & (hsv[:, :, 2] >= shadow_threshold)

    result = {
        "color_valid_pixels": int(np.count_nonzero(valid)),
        "shadow_threshold_v": int(shadow_threshold),
    }
    for color_name, color_image, channels in (
        ("hsv", hsv, ("h", "s", "v")),
        ("lab", lab, ("l", "a", "b")),
    ):
        pixels = color_image[valid]
        for index, channel in enumerate(channels):
            if pixels.size == 0:
                mean = std = 0.0
            else:
                mean = float(np.mean(pixels[:, index]))
                std = float(np.std(pixels[:, index]))
            result[f"{color_name}_{channel}_mean"] = round(mean, 4)
            result[f"{color_name}_{channel}_std"] = round(std, 4)

    red = (
        valid
        & ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 50)
    )
    denominator = max(1, int(np.count_nonzero(valid)))
    result["red_pixel_count"] = int(np.count_nonzero(red))
    result["red_ratio_robust"] = round(result["red_pixel_count"] / denominator, 6)
    result["valid_mask"] = np.where(valid, 255, 0).astype(np.uint8)
    result["red_mask"] = np.where(red, 255, 0).astype(np.uint8)
    return result
