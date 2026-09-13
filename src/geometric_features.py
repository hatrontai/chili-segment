"""Geometric mesocarp-thickness and outer-mesocarp wrinkle measurements.

The implementation in this module is the runtime counterpart of
``wrinkle_thickness_experiment.ipynb``.  Distances are expressed in millimetres
through the QR-derived (or explicitly documented fallback) mm/pixel ratio.
"""

import math

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


MESOCARP_WRINKLE_SEGMENT_THRESHOLD_MM = 5.0
MESOCARP_THICKNESS_SAMPLE_SPACING_MM = 1.0
SKELETON_CLOSING_MM = 1.4
SKELETON_OPENING_MM = 0.6


def _as_bool(mask):
    return np.asarray(mask) > 0


def build_skeleton_graph(coords, shape):
    """Build a weighted 8-neighbour graph without redundant diagonals."""
    coords = np.asarray(coords, dtype=int)
    node_count = len(coords)
    index_image = np.full(shape, -1, dtype=np.int32)
    index_image[coords[:, 0], coords[:, 1]] = np.arange(node_count)
    rows, cols, weights = [], [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        shifted_y = coords[:, 0] + dy
        shifted_x = coords[:, 1] + dx
        valid = (
            (shifted_y >= 0)
            & (shifted_y < shape[0])
            & (shifted_x >= 0)
            & (shifted_x < shape[1])
        )
        source = np.flatnonzero(valid)
        target = index_image[shifted_y[valid], shifted_x[valid]]
        connected = target >= 0
        source, target = source[connected], target[connected]
        if dy != 0 and dx != 0 and len(source):
            sy, sx = coords[source, 0], coords[source, 1]
            has_orthogonal = (index_image[sy, sx + dx] >= 0) | (
                index_image[sy + dy, sx] >= 0
            )
            source, target = source[~has_orthogonal], target[~has_orthogonal]
        edge_weight = math.hypot(dy, dx)
        rows.extend(source.tolist() + target.tolist())
        cols.extend(target.tolist() + source.tolist())
        weights.extend([edge_weight] * (2 * len(source)))
    return csr_matrix((weights, (rows, cols)), shape=(node_count, node_count))


def reconstruct_graph_path(predecessors, source, target):
    path = [int(target)]
    current = int(target)
    while current != int(source):
        current = int(predecessors[current])
        if current < 0:
            return []
        path.append(current)
    path.reverse()
    return path


def longest_backbone_skeleton(skeleton):
    """Keep the longest endpoint-to-endpoint geodesic in branched components."""
    skeleton = np.asarray(skeleton, dtype=bool)
    output = np.zeros_like(skeleton)
    labels, component_count = ndi.label(skeleton, structure=np.ones((3, 3), int))
    for component_id in range(1, component_count + 1):
        coords = np.argwhere(labels == component_id)
        if len(coords) <= 2:
            output[coords[:, 0], coords[:, 1]] = True
            continue

        graph = build_skeleton_graph(coords, skeleton.shape)
        degree = np.diff(graph.indptr)
        endpoints = np.flatnonzero(degree == 1)
        if not np.any(degree >= 3):
            output[coords[:, 0], coords[:, 1]] = True
            continue
        if len(endpoints) < 2:
            seed_distances = dijkstra(graph, directed=False, indices=0)
            first = int(
                np.argmax(np.where(np.isfinite(seed_distances), seed_distances, -np.inf))
            )
            first_distances = dijkstra(graph, directed=False, indices=first)
            second = int(
                np.argmax(np.where(np.isfinite(first_distances), first_distances, -np.inf))
            )
            endpoints = np.array([first, second], dtype=int)

        distances, predecessors = dijkstra(
            graph, directed=False, indices=endpoints, return_predecessors=True
        )
        endpoint_distances = distances[:, endpoints]
        endpoint_distances[~np.isfinite(endpoint_distances)] = -np.inf
        source_pos, target_pos = np.unravel_index(
            np.argmax(endpoint_distances), endpoint_distances.shape
        )
        source_node = int(endpoints[source_pos])
        target_node = int(endpoints[target_pos])
        path = reconstruct_graph_path(
            predecessors[source_pos], source_node, target_node
        )
        if not path:
            continue
        path_coords = coords[np.asarray(path, dtype=int)]
        output[path_coords[:, 0], path_coords[:, 1]] = True
    return output


def smooth_mesocarp_for_skeleton(
    mesocarp,
    ratio,
    closing_mm=SKELETON_CLOSING_MM,
    opening_mm=SKELETON_OPENING_MM,
):
    """Smooth only the skeleton source; measurements still use raw mesocarp."""
    closing_size_px = max(3, int(round(closing_mm / ratio)))
    opening_size_px = max(2, int(round(opening_mm / ratio)))
    closed = ndi.binary_closing(
        mesocarp, structure=np.ones((closing_size_px, closing_size_px))
    )
    smoothed = ndi.binary_opening(
        closed, structure=np.ones((opening_size_px, opening_size_px))
    )
    labels, count = ndi.label(smoothed)
    if count:
        sizes = ndi.sum(smoothed, labels, range(1, count + 1))
        keep = [index + 1 for index, size in enumerate(sizes) if size >= 30]
        smoothed = np.isin(labels, keep)
    return smoothed


def build_mesocarp_data(chili, ratio):
    """Create mesocarp, its inner/outer references and measurement skeleton."""
    if ratio <= 0:
        raise ValueError("ratio must be greater than zero")

    chili_mask = _as_bool(chili.mask_chili)
    endocarp = _as_bool(chili.mask_endocarp)
    outer = _as_bool(chili.mask_outer)
    pcp = _as_bool(chili.mask_pcp)
    seed = _as_bool(chili.mask_seed)
    pedicel = _as_bool(chili.mask_pedicel)

    mesocarp = _as_bool(chili.mask_mesocarp)
    labels, count = ndi.label(mesocarp)
    if count:
        sizes = ndi.sum(mesocarp, labels, range(1, count + 1))
        keep = [index + 1 for index, size in enumerate(sizes) if size >= 30]
        mesocarp = np.isin(labels, keep)

    boundary = mesocarp & ~ndi.binary_erosion(mesocarp, structure=np.ones((3, 3)))
    inner_reference = endocarp | pcp | seed
    outer_reference = outer | ~chili_mask
    distance_to_inner = ndi.distance_transform_edt(~inner_reference)
    distance_to_outer = ndi.distance_transform_edt(~outer_reference)
    inner_edge = boundary & (distance_to_inner <= distance_to_outer)
    outer_edge = boundary & (distance_to_outer < distance_to_inner)

    # This is raster tolerance around the pedicel, not the wrinkle threshold.
    pedicel_exclusion = ndi.binary_dilation(
        pedicel, structure=np.ones((3, 3)), iterations=2
    )
    signed_distance = ndi.distance_transform_edt(mesocarp) - ndi.distance_transform_edt(
        ~mesocarp
    )

    raw_skeleton = skeletonize(mesocarp)
    mesocarp_for_skeleton = smooth_mesocarp_for_skeleton(mesocarp, ratio)
    smoothed_skeleton = skeletonize(mesocarp_for_skeleton)
    minimum_pixels = max(20, int(0.50 * np.count_nonzero(raw_skeleton)))
    use_smoothed = np.count_nonzero(smoothed_skeleton) >= minimum_pixels
    source_skeleton = smoothed_skeleton if use_smoothed else raw_skeleton
    skeleton = longest_backbone_skeleton(source_skeleton)
    measurement_skeleton = skeleton & mesocarp
    if np.count_nonzero(measurement_skeleton) < 20:
        source_skeleton = raw_skeleton
        skeleton = longest_backbone_skeleton(raw_skeleton)
        measurement_skeleton = skeleton
        use_smoothed = False

    return {
        "mesocarp": mesocarp,
        "boundary": boundary,
        "inner_reference": inner_reference,
        "outer_reference": outer_reference,
        "distance_to_inner": distance_to_inner,
        "distance_to_outer": distance_to_outer,
        "inner_edge": inner_edge,
        "outer_edge": outer_edge,
        "pedicel_exclusion": pedicel_exclusion,
        "wrinkle_outer_mesocarp_edge": outer_edge & ~pedicel_exclusion,
        "signed_distance": signed_distance,
        "mesocarp_for_skeleton": mesocarp_for_skeleton,
        "skeleton": skeleton,
        "measurement_skeleton": measurement_skeleton,
        "skeleton_source": "smoothed_mesocarp" if use_smoothed else "raw_mesocarp",
    }


def bilinear_many(array, x, y, outside_value=-1.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    result = np.full(np.broadcast(x, y).shape, float(outside_value))
    height, width = array.shape
    valid = (x >= 0) & (y >= 0) & (x <= width - 1) & (y <= height - 1)
    if not np.any(valid):
        return result
    xv, yv = x[valid], y[valid]
    x0, y0 = np.floor(xv).astype(int), np.floor(yv).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = xv - x0, yv - y0
    result[valid] = (
        array[y0, x0] * (1 - dx) * (1 - dy)
        + array[y0, x1] * dx * (1 - dy)
        + array[y1, x0] * (1 - dx) * dy
        + array[y1, x1] * dx * dy
    )
    return result


def bilinear(array, x, y, outside_value=-1.0):
    return float(
        bilinear_many(array, np.array([x]), np.array([y]), outside_value)[0]
    )


def ray_exits(center_xy, directions, signed_distance):
    """Cast multiple rays and interpolate their signed-distance zero crossing."""
    directions = np.asarray(directions, float)
    count = len(directions)
    height, width = signed_distance.shape
    max_distance = float(np.hypot(height, width))
    distances = np.zeros(count)
    values = np.full(
        count, bilinear(signed_distance, center_xy[0], center_xy[1])
    )
    active = values > 0
    low, high = np.full(count, np.nan), np.full(count, np.nan)
    for _ in range(int(max_distance / 0.35) + 2):
        if not np.any(active):
            break
        ids = np.flatnonzero(active)
        steps = np.clip(values[ids] * 0.65, 0.35, 3.0)
        next_distances = np.minimum(max_distance, distances[ids] + steps)
        points = np.asarray(center_xy) + next_distances[:, None] * directions[ids]
        next_values = bilinear_many(
            signed_distance, points[:, 0], points[:, 1]
        )
        hit = next_values <= 0
        hit_ids = ids[hit]
        low[hit_ids], high[hit_ids] = distances[hit_ids], next_distances[hit]
        active[hit_ids] = False
        remaining = ids[~hit]
        distances[remaining] = next_distances[~hit]
        values[remaining] = next_values[~hit]
        active[remaining[distances[remaining] >= max_distance]] = False

    valid = np.isfinite(low)
    roots = np.full(count, np.nan)
    if np.any(valid):
        ids = np.flatnonzero(valid)
        lo, hi = low[ids].copy(), high[ids].copy()
        for _ in range(14):
            mid = (lo + hi) / 2
            points = np.asarray(center_xy) + mid[:, None] * directions[ids]
            mid_values = bilinear_many(
                signed_distance, points[:, 0], points[:, 1]
            )
            inside = mid_values > 0
            lo[inside], hi[~inside] = mid[inside], mid[~inside]
        roots[ids] = (lo + hi) / 2
    return np.asarray(center_xy) + roots[:, None] * directions, roots, valid


def sections_for_angles(center_xy, angles, mesocarp_data):
    angles = np.asarray(angles, float)
    radians = np.deg2rad(angles)
    directions = np.column_stack([np.cos(radians), np.sin(radians)])
    p_pos, r_pos, ok_pos = ray_exits(
        center_xy, directions, mesocarp_data["signed_distance"]
    )
    p_neg, r_neg, ok_neg = ray_exits(
        center_xy, -directions, mesocarp_data["signed_distance"]
    )
    valid = ok_pos & ok_neg
    pos_inner = bilinear_many(
        mesocarp_data["distance_to_inner"], p_pos[:, 0], p_pos[:, 1], np.inf
    ) <= bilinear_many(
        mesocarp_data["distance_to_outer"], p_pos[:, 0], p_pos[:, 1], np.inf
    )
    neg_inner = bilinear_many(
        mesocarp_data["distance_to_inner"], p_neg[:, 0], p_neg[:, 1], np.inf
    ) <= bilinear_many(
        mesocarp_data["distance_to_outer"], p_neg[:, 0], p_neg[:, 1], np.inf
    )
    valid &= pos_inner != neg_inner
    output = []
    for index in np.flatnonzero(valid):
        inner, outer = (
            (p_pos[index], p_neg[index])
            if pos_inner[index]
            else (p_neg[index], p_pos[index])
        )
        output.append(
            {
                "inner": inner,
                "outer": outer,
                "length_px": float(r_pos[index] + r_neg[index]),
                "angle_deg": float(angles[index] % 180),
            }
        )
    return output


def measure_skeleton_point(center_xy, mesocarp_data):
    coarse = sections_for_angles(
        center_xy, np.arange(0.0, 180.0, 1.0), mesocarp_data
    )
    if not coarse:
        return None
    best = min(coarse, key=lambda item: item["length_px"])
    fine_angles = np.arange(
        best["angle_deg"] - 1.0, best["angle_deg"] + 1.0001, 0.1
    )
    refined = sections_for_angles(center_xy, fine_angles, mesocarp_data)
    return min(refined or [best], key=lambda item: item["length_px"])


def ordered_skeleton_paths(skeleton):
    """Convert each unbranched component to an ordered polyline."""
    skeleton = np.asarray(skeleton, dtype=bool)
    labels, component_count = ndi.label(skeleton, structure=np.ones((3, 3), int))
    paths = []
    for component_id in range(1, component_count + 1):
        coords = np.argwhere(labels == component_id)
        if len(coords) == 1:
            paths.append(
                {
                    "component_id": component_id,
                    "points_xy": coords[:, ::-1].astype(float),
                    "closed": False,
                }
            )
            continue
        graph = build_skeleton_graph(coords, skeleton.shape)
        degree = np.diff(graph.indptr)
        endpoints = np.flatnonzero(degree == 1)
        closed = len(endpoints) == 0 and np.all(degree == 2)

        if closed:
            start = 0
            path, visited = [start], {start}
            previous, current = -1, start
            while True:
                neighbours = graph.indices[
                    graph.indptr[current] : graph.indptr[current + 1]
                ]
                candidates = [
                    int(node) for node in neighbours if int(node) != previous
                ]
                if previous < 0:
                    candidates.sort(
                        key=lambda node: (coords[node, 0], coords[node, 1])
                    )
                next_node = next(
                    (node for node in candidates if node not in visited), start
                )
                if next_node == start:
                    break
                path.append(next_node)
                visited.add(next_node)
                previous, current = current, next_node
            if len(path) != len(coords):
                continue
        else:
            if len(endpoints) < 2:
                seed_distance = dijkstra(graph, directed=False, indices=0)
                source = int(
                    np.argmax(
                        np.where(np.isfinite(seed_distance), seed_distance, -np.inf)
                    )
                )
            else:
                source = min(
                    endpoints, key=lambda node: (coords[node, 0], coords[node, 1])
                )
            distances, predecessors = dijkstra(
                graph, directed=False, indices=int(source), return_predecessors=True
            )
            target_candidates = endpoints if len(endpoints) >= 2 else np.arange(len(coords))
            target = int(target_candidates[np.argmax(distances[target_candidates])])
            path = reconstruct_graph_path(predecessors, source, target)
            if not path:
                continue

        paths.append(
            {
                "component_id": component_id,
                "points_xy": coords[np.asarray(path, dtype=int)][:, ::-1].astype(float),
                "closed": closed,
            }
        )
    return paths


def sample_skeleton_at_spacing(
    skeleton, ratio, spacing_mm=MESOCARP_THICKNESS_SAMPLE_SPACING_MM
):
    """Interpolate samples by path length and preserve open-path endpoints."""
    if ratio <= 0 or spacing_mm <= 0:
        raise ValueError("ratio and spacing_mm must be greater than zero")
    spacing_px = spacing_mm / ratio
    rows, sample_id = [], 0
    for path_info in ordered_skeleton_paths(skeleton):
        points = path_info["points_xy"]
        closed = path_info["closed"]
        interpolation_points = np.vstack([points, points[0]]) if closed else points
        lengths = np.linalg.norm(np.diff(interpolation_points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total_px = float(cumulative[-1])
        if total_px <= 1e-9:
            positions = np.array([0.0])
        elif closed:
            positions = np.arange(0.0, total_px - 1e-9, spacing_px)
        else:
            positions = np.arange(0.0, total_px + 1e-9, spacing_px)
            if total_px - positions[-1] > 1e-9:
                positions = np.append(positions, total_px)

        for local_index, position in enumerate(positions):
            if len(lengths) == 0:
                point = points[0]
            else:
                segment_index = min(
                    np.searchsorted(cumulative, position, side="right") - 1,
                    len(lengths) - 1,
                )
                segment_length = max(lengths[segment_index], 1e-12)
                alpha = (position - cumulative[segment_index]) / segment_length
                point = interpolation_points[segment_index] + alpha * (
                    interpolation_points[segment_index + 1]
                    - interpolation_points[segment_index]
                )
            if local_index == 0:
                previous_gap_mm = (
                    (total_px - positions[-1]) * ratio if closed else np.nan
                )
                is_tail_gap = bool(
                    closed and previous_gap_mm < spacing_mm - 1e-9
                )
            else:
                previous_gap_mm = (
                    positions[local_index] - positions[local_index - 1]
                ) * ratio
                is_tail_gap = bool(
                    not closed
                    and local_index == len(positions) - 1
                    and previous_gap_mm < spacing_mm - 1e-9
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "component_id": path_info["component_id"],
                    "sample_index": local_index,
                    "path_position_mm": position * ratio,
                    "skeleton_x": float(point[0]),
                    "skeleton_y": float(point[1]),
                    "is_endpoint": bool(
                        not closed and local_index in (0, len(positions) - 1)
                    ),
                    "is_closed_component": bool(closed),
                    "previous_gap_mm": previous_gap_mm,
                    "is_tail_gap": is_tail_gap,
                }
            )
            sample_id += 1
    return pd.DataFrame(rows)


def measure_mesocarp_thickness(mesocarp_data, ratio):
    samples = sample_skeleton_at_spacing(
        mesocarp_data["measurement_skeleton"], ratio
    )
    rows = []
    for _, sample in samples.iterrows():
        row = sample.to_dict()
        result = measure_skeleton_point(
            (sample.skeleton_x, sample.skeleton_y), mesocarp_data
        )
        if result is None:
            row["measurement_status"] = "ray_failed"
        else:
            row.update(
                {
                    "inner_x": result["inner"][0],
                    "inner_y": result["inner"][1],
                    "outer_x": result["outer"][0],
                    "outer_y": result["outer"][1],
                    "angle_deg": result["angle_deg"],
                    "thickness_px": result["length_px"],
                    "thickness_mm": result["length_px"] * ratio,
                    "measurement_status": "measured",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def skeleton_branch_mask(skeleton):
    skeleton = np.asarray(skeleton, dtype=bool)
    neighbour_count = ndi.convolve(
        skeleton.astype(int), np.ones((3, 3), dtype=int), mode="constant", cval=0
    ) - skeleton.astype(int)
    return skeleton & (neighbour_count >= 3)


def skeleton_branch_distance(skeleton):
    branch_points = skeleton_branch_mask(skeleton)
    if not np.any(branch_points):
        return np.full(skeleton.shape, np.inf)
    return ndi.distance_transform_edt(~branch_points)


def filter_clean_thickness_points(
    points,
    mesocarp_data,
    ratio,
    k=8,
    thickness_mad_multiplier=3.0,
    angle_mad_multiplier=2.0,
    pedicel_margin_mm=1.5,
    branch_margin_mm=2.5,
):
    if len(points) < 3:
        return points.copy().reset_index(drop=True)

    pedicel_distance = ndi.distance_transform_edt(
        ~mesocarp_data["pedicel_exclusion"]
    )
    near_pedicel = bilinear_many(
        pedicel_distance,
        points.skeleton_x.to_numpy(),
        points.skeleton_y.to_numpy(),
    ) < (pedicel_margin_mm / ratio)
    branch_distance = skeleton_branch_distance(
        mesocarp_data["measurement_skeleton"]
    )
    if np.isinf(branch_distance).all():
        near_branch = np.zeros(len(points), dtype=bool)
    else:
        near_branch = bilinear_many(
            branch_distance,
            points.skeleton_x.to_numpy(),
            points.skeleton_y.to_numpy(),
            np.inf,
        ) < (branch_margin_mm / ratio)

    coordinates = points[["skeleton_x", "skeleton_y"]].to_numpy()
    neighbour_k = min(max(3, k + 1), len(coordinates))
    _, neighbour_index = cKDTree(coordinates).query(coordinates, k=neighbour_k)
    neighbour_index = np.asarray(neighbour_index)[:, 1:]

    thickness = points.thickness_mm.to_numpy()
    local_median = np.median(thickness[neighbour_index], axis=1)
    local_mad = np.maximum(
        np.median(
            np.abs(thickness[neighbour_index] - local_median[:, None]), axis=1
        ),
        0.05,
    )
    thickness_outlier = (
        np.abs(thickness - local_median) / local_mad > thickness_mad_multiplier
    )

    angles = points.angle_deg.to_numpy()
    neighbour_angles = angles[neighbour_index]
    doubled = np.deg2rad(2.0 * neighbour_angles)
    local_angle = (
        0.5
        * np.rad2deg(
            np.arctan2(
                np.mean(np.sin(doubled), axis=1),
                np.mean(np.cos(doubled), axis=1),
            )
        )
    ) % 180.0
    angle_deviation = np.abs(angles - local_angle)
    angle_deviation = np.minimum(angle_deviation, 180.0 - angle_deviation)
    neighbour_deviation = np.abs(neighbour_angles - local_angle[:, None])
    neighbour_deviation = np.minimum(
        neighbour_deviation, 180.0 - neighbour_deviation
    )
    local_angle_median = np.median(neighbour_deviation, axis=1)
    local_angle_mad = np.maximum(
        np.median(
            np.abs(neighbour_deviation - local_angle_median[:, None]), axis=1
        ),
        1.0,
    )
    angle_outlier = angle_deviation > (
        local_angle_median + angle_mad_multiplier * local_angle_mad
    )
    return points[
        (~near_pedicel) & (~near_branch) & (~thickness_outlier) & (~angle_outlier)
    ].reset_index(drop=True)


def calculate_mesocarp_thickness_stats(mesocarp_data, ratio):
    """Return min/average/max mesocarp thickness from accepted 1 mm samples."""
    measured = measure_mesocarp_thickness(mesocarp_data, ratio)
    if measured.empty or "measurement_status" not in measured:
        return {
            "min_mesocarp_thickness": None,
            "avg_mesocarp_thickness": None,
            "max_mesocarp_thickness": None,
        }
    valid = measured[measured.measurement_status == "measured"].reset_index(drop=True)
    if valid.empty:
        return {
            "min_mesocarp_thickness": None,
            "avg_mesocarp_thickness": None,
            "max_mesocarp_thickness": None,
        }
    clean = filter_clean_thickness_points(valid, mesocarp_data, ratio)
    minimum_clean_count = max(20, int(0.30 * len(valid)))
    used = clean if len(clean) >= minimum_clean_count else valid
    return {
        "min_mesocarp_thickness": round(float(used.thickness_mm.min()), 2),
        "avg_mesocarp_thickness": round(float(used.thickness_mm.mean()), 2),
        "max_mesocarp_thickness": round(float(used.thickness_mm.max()), 2),
    }


def smooth_circular_labels(labels, window=7):
    labels = np.asarray(labels, dtype=np.uint8)
    if len(labels) < window:
        return labels.astype(bool)
    radius = window // 2
    padded = np.concatenate([labels[-radius:], labels, labels[:radius]])
    votes = np.convolve(padded, np.ones(window, dtype=int), mode="valid")
    return votes >= (window // 2 + 1)


def split_true_runs(points, selected, closed):
    points = np.asarray(points, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if not np.any(selected):
        return []
    if np.all(selected):
        return [np.vstack([points, points[0]]) if closed else points]
    if closed:
        false_index = int(np.flatnonzero(~selected)[0])
        order = np.r_[
            np.arange(false_index + 1, len(points)), np.arange(0, false_index + 1)
        ]
        points, selected = points[order], selected[order]
    runs, current = [], []
    for point, keep in zip(points, selected):
        if keep:
            current.append(point)
        elif current:
            if len(current) >= 2:
                runs.append(np.asarray(current, dtype=float))
            current = []
    if len(current) >= 2:
        runs.append(np.asarray(current, dtype=float))
    return runs


def extract_outer_mesocarp_edge_runs(mesocarp_data, minimum_arc_px=5.0):
    contours, _ = cv2.findContours(
        mesocarp_data["mesocarp"].astype(np.uint8) * 255,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    runs = []
    for contour in contours:
        points = contour[:, 0, :].astype(float)
        if len(points) < 3:
            continue
        labels, outside_pedicel_flags = [], []
        for x, y in points:
            distance_inner = bilinear(
                mesocarp_data["distance_to_inner"], x, y, np.inf
            )
            distance_outer = bilinear(
                mesocarp_data["distance_to_outer"], x, y, np.inf
            )
            yi = int(
                np.clip(round(y), 0, mesocarp_data["mesocarp"].shape[0] - 1)
            )
            xi = int(
                np.clip(round(x), 0, mesocarp_data["mesocarp"].shape[1] - 1)
            )
            labels.append(distance_outer < distance_inner)
            outside_pedicel_flags.append(
                not mesocarp_data["pedicel_exclusion"][yi, xi]
            )
        selected = smooth_circular_labels(labels, window=7)
        selected &= np.asarray(outside_pedicel_flags, dtype=bool)
        for run in split_true_runs(points, selected, closed=True):
            arc = np.linalg.norm(np.diff(run, axis=0), axis=1).sum()
            if arc >= minimum_arc_px:
                runs.append(run)
    return sorted(
        runs,
        key=lambda run: np.linalg.norm(np.diff(run, axis=0), axis=1).sum(),
        reverse=True,
    )


def polyline_geometry(points):
    points = np.asarray(points, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, segment_lengths > 1e-12]]
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return points, np.r_[0.0, np.cumsum(segment_lengths)]


def point_at_arc(points, cumulative, arc_position):
    position = float(np.clip(arc_position, 0.0, cumulative[-1]))
    if position >= cumulative[-1]:
        return points[-1].copy()
    index = int(np.searchsorted(cumulative, position, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)
    length = cumulative[index + 1] - cumulative[index]
    alpha = 0.0 if length <= 1e-12 else (position - cumulative[index]) / length
    return points[index] + alpha * (points[index + 1] - points[index])


def first_circle_intersection(points, cumulative, start_s, radius):
    center = point_at_arc(points, cumulative, start_s)
    index = int(np.searchsorted(cumulative, start_s, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)
    for segment_index in range(index, len(points) - 1):
        a = (
            point_at_arc(points, cumulative, start_s)
            if segment_index == index
            else points[segment_index]
        )
        a_s = start_s if segment_index == index else cumulative[segment_index]
        b = points[segment_index + 1]
        vector = b - a
        coefficient_a = float(np.dot(vector, vector))
        if coefficient_a <= 1e-12:
            continue
        relative = a - center
        coefficient_b = 2.0 * float(np.dot(relative, vector))
        coefficient_c = float(np.dot(relative, relative) - radius * radius)
        discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
        if discriminant < 0:
            continue
        root_discriminant = math.sqrt(max(0.0, discriminant))
        roots = sorted(
            (
                (-coefficient_b - root_discriminant) / (2.0 * coefficient_a),
                (-coefficient_b + root_discriminant) / (2.0 * coefficient_a),
            )
        )
        for root in roots:
            if -1e-10 <= root <= 1.0 + 1e-10:
                root = float(np.clip(root, 0.0, 1.0))
                candidate_s = a_s + root * math.sqrt(coefficient_a)
                if candidate_s > start_s + 1e-8:
                    return candidate_s
    return None


def balanced_tail_split(points, cumulative, start_s, end_s, radius):
    start = point_at_arc(points, cumulative, start_s)
    end = point_at_arc(points, cumulative, end_s)
    margin = max(1e-7, (end_s - start_s) * 1e-6)

    def objective(split_s):
        split = point_at_arc(points, cumulative, split_s)
        first = np.linalg.norm(split - start)
        second = np.linalg.norm(end - split)
        overflow = max(0.0, first - radius) + max(0.0, second - radius)
        return abs(first - second) + 1000.0 * overflow

    result = minimize_scalar(
        objective,
        bounds=(start_s + margin, end_s - margin),
        method="bounded",
        options={"xatol": 1e-7},
    )
    split_s = float(result.x)
    split = point_at_arc(points, cumulative, split_s)
    chords = (np.linalg.norm(split - start), np.linalg.norm(end - split))
    if max(chords) > radius + 1e-4:
        candidates = []
        for value in np.linspace(start_s + margin, end_s - margin, 2001):
            point = point_at_arc(points, cumulative, value)
            first = np.linalg.norm(point - start)
            second = np.linalg.norm(end - point)
            if first <= radius + 1e-6 and second <= radius + 1e-6:
                candidates.append((abs(first - second), value))
        if candidates:
            split_s = min(candidates)[1]
    return split_s


def segment_polyline_by_exact_chord(points, chord_px):
    if chord_px <= 0:
        raise ValueError("chord_px must be greater than zero")
    points, cumulative = polyline_geometry(points)
    total = cumulative[-1]
    if total <= 1e-10:
        return points, cumulative, []
    cuts = [0.0]
    while True:
        next_s = first_circle_intersection(points, cumulative, cuts[-1], chord_px)
        if next_s is None or total - next_s <= 1e-8:
            break
        cuts.append(next_s)
    if len(cuts) == 1:
        cuts.append(total)
    else:
        if total - cuts[-1] > 1e-8:
            previous_start = cuts[-2] if len(cuts) >= 2 else 0.0
            split_s = balanced_tail_split(
                points, cumulative, previous_start, total, chord_px
            )
            cuts = cuts[:-1] + [split_s, total]
        elif cuts[-1] < total:
            cuts.append(total)
    if cuts[-1] < total - 1e-8:
        cuts.append(total)

    segments = []
    for start_s, end_s in zip(cuts[:-1], cuts[1:]):
        start = point_at_arc(points, cumulative, start_s)
        end = point_at_arc(points, cumulative, end_s)
        chord = float(np.linalg.norm(end - start))
        arc = float(end_s - start_s)
        if chord <= 1e-10:
            continue
        segments.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "start": start,
                "end": end,
                "arc_px": arc,
                "chord_px": chord,
                "curvature_ratio": arc / chord,
                "curvature_excess": arc / chord - 1.0,
            }
        )
    return points, cumulative, segments


def calculate_mesocarp_wrinkle_index(
    mesocarp_data,
    ratio,
    mesocarp_wrinkle_segment_threshold_mm=MESOCARP_WRINKLE_SEGMENT_THRESHOLD_MM,
):
    """Return mean outer-mesocarp curvature excess, excluding the pedicel."""
    if ratio <= 0 or mesocarp_wrinkle_segment_threshold_mm <= 0:
        raise ValueError(
            "ratio and mesocarp_wrinkle_segment_threshold_mm must be greater than zero"
        )
    chord_px = mesocarp_wrinkle_segment_threshold_mm / ratio
    excesses = []
    for run in extract_outer_mesocarp_edge_runs(mesocarp_data):
        _, _, segments = segment_polyline_by_exact_chord(run, chord_px)
        excesses.extend(segment["curvature_excess"] for segment in segments)
    if not excesses:
        return None
    return round(float(np.mean(excesses)), 5)
