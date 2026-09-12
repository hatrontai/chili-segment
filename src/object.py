import itertools
import math
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from pyzbar.pyzbar import decode
except Exception:
    decode = None
    print("Warning: pyzbar not installed correctly. QR code scaling disabled.")
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as sk_watershed
from src.geometric_features import (
    WRINKLE_SEGMENT_THRESHOLD_MM,
    build_wall_data,
    calculate_thickness_stats,
    calculate_wrinkle,
)

CLASS_NAMES = {0: "seed", 1: "outer", 2: "placenta", 3: "endocarp", 4: "chili"}
CLASS_COLORS = {
    0: (125, 0, 25),
    1: (0, 155, 0),
    2: (55, 0, 155),
    3: (0, 155, 155),
    4: (155, 0, 155),
}

MAX_STEM_RATIO = 0.35
MIN_STEM_RATIO = 0.02

PLACENTA_MIN_COMPONENT_AREA = 30

CORNER_TOP_REGION_FRAC = 0.55
CORNER_ANGLE_WINDOW = 12
CORNER_ANGLE_MIN_DEG = 40
CORNER_ANGLE_MAX_DEG = 130
CORNER_MAX_Y_DIST_FRAC = 0.18
CORNER_MIN_X_SEP_FRAC = 0.10
CORNER_MIN_RUN_LEN = 1

CLOSE_LINE_THICKNESS = 3
MIN_FLESH_COMPONENT_AREA = 30

INNER_REF_DILATE_PX_START = 6
INNER_REF_DILATE_PX_MAX = 30
INNER_REF_DILATE_PX_STEP = 3
MIN_INNER_RUN_LEN_FRAC_START = 0.05
MIN_INNER_RUN_LEN_FRAC_FLOOR = 0.003

RUN_LEN_SHRINK_FACTOR = 0.6
RUN_LEN_MAX_ATTEMPTS = 6
MIN_TIP_X_SEPARATION_FRAC = 0.15

TIP_CURVATURE_WINDOW = 6
TIP_CURVATURE_WEIGHT = 1.0
TIP_DIST_PENALTY_WEIGHT = 0.5

CHILI_MIN_COMPONENT_AREA = 30
CHILI_MIN_COMPONENT_RATIO = 0.002

# Arithmetic mean of the QR ratios from the 22 bundled dataset images.  It is
# an estimated scale only and is always accompanied by ``scale_source``.
DEFAULT_DATASET_RATIO_MM_PER_PX = 0.094305839


def find_centroid(matrix):
    if matrix is None or matrix.size == 0:
        return None
    y_coords, x_coords = np.where(matrix > 0)
    if len(x_coords) == 0:
        return None
    cx, cy = np.mean(x_coords), np.mean(y_coords)
    return cx, cy


def calculate_angle(p1, p2, p3):
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    norm_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if norm_v1 * norm_v2 == 0:
        return 0, norm_v1, norm_v2
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    cos_val = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle = math.acos(cos_val)
    return angle, norm_v1, norm_v2


def korean_imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype=dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def draw_full_mask_with_labels(image, objects, alpha=0.5):
    overlay = image.copy()
    output = image.copy()
    objects_sorted = sorted(objects, key=lambda x: x["class_id"], reverse=True)
    for obj in objects_sorted:
        pts = np.array(obj["points"], dtype=np.float32)
        pts = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        color = CLASS_COLORS[obj["class_id"]]
        cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    objects_outline_sorted = sorted(objects, key=lambda x: x["class_id"])
    for idx, obj in enumerate(objects_outline_sorted):
        pts = np.array(obj["points"], dtype=np.float32)
        pts = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        color = CLASS_COLORS[obj["class_id"]]
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2)
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = pts[0][0]
        label = f"{obj['class_name']}"
        cv2.putText(
            output,
            label,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def get_placenta_contour(mask_placenta, min_area=PLACENTA_MIN_COMPONENT_AREA):
    contours, _ = cv2.findContours(
        mask_placenta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        valid = contours
    c = min(valid, key=lambda cc: cc.reshape(-1, 2)[:, 1].min())
    return c.reshape(-1, 2)


def build_stem_mask_from_corners(mask_shape, left_chain, i_left, right_chain, i_right):
    left_part = left_chain[0 : i_left + 1]
    right_part = right_chain[0 : i_right + 1][::-1]
    polygon = np.concatenate([left_part, right_part], axis=0).astype(np.int32)
    mask_stem = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask_stem, [polygon.reshape(-1, 1, 2)], 255)
    return mask_stem, polygon


def get_apex_point(mask_placenta):
    ys_p, xs_p = np.where(mask_placenta > 0)
    apex_idx = int(np.argmin(ys_p))
    return (int(xs_p[apex_idx]), int(ys_p[apex_idx]))


def _contour_orientation_sign(contour):
    area = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.int32), oriented=True)
    return 1.0 if area >= 0 else -1.0


def split_into_side_chains(contour):
    ys = contour[:, 1]
    n = len(contour)
    apex_idx = int(np.argmin(ys))
    bottom_idx = int(np.argmax(ys))
    idxA = np.arange(apex_idx, apex_idx + ((bottom_idx - apex_idx) % n) + 1) % n
    idxB_raw = np.arange(bottom_idx, bottom_idx + ((apex_idx - bottom_idx) % n) + 1) % n
    idxB = idxB_raw[::-1]
    chainA, chainB = contour[idxA], contour[idxB]
    if chainA[:, 0].mean() <= chainB[:, 0].mean():
        return chainA, idxA, chainB, idxB
    return chainB, idxB, chainA, idxA


def compute_angle_and_concavity_profile(contour, window):
    n = len(contour)
    orient = _contour_orientation_sign(contour)
    angle_deg = np.full(n, 180.0, dtype=np.float64)
    concave = np.zeros(n, dtype=bool)
    for i in range(n):
        p1 = contour[(i - window) % n]
        p2 = contour[i]
        p3 = contour[(i + window) % n]
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        angle_rad, n1, n2 = calculate_angle(p1, p2, p3)
        if n1 == 0 or n2 == 0:
            continue
        angle_deg[i] = math.degrees(angle_rad)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sign_cross = 1.0 if cross >= 0 else -1.0
        concave[i] = sign_cross != orient
    return angle_deg, concave


def find_armpit_points(
    contour,
    top_region_frac=CORNER_TOP_REGION_FRAC,
    window=CORNER_ANGLE_WINDOW,
    angle_min=CORNER_ANGLE_MIN_DEG,
    angle_max=CORNER_ANGLE_MAX_DEG,
    max_y_dist_frac=CORNER_MAX_Y_DIST_FRAC,
    min_x_sep_frac=CORNER_MIN_X_SEP_FRAC,
):
    xs, ys = contour[:, 0], contour[:, 1]
    y_min, y_max = ys.min(), ys.max()
    height = max(y_max - y_min, 1)
    width = max(xs.max() - xs.min(), 1)
    y_limit = y_min + height * top_region_frac
    angle_deg, concave = compute_angle_and_concavity_profile(contour, window)
    left_chain, left_idx, right_chain, right_idx = split_into_side_chains(contour)
    debug_candidates = []

    def collect_side_candidates(chain, idx_map, min_run=CORNER_MIN_RUN_LEN):
        ys_c = chain[:, 1]
        region_local = np.where(ys_c <= y_limit)[0]
        if len(region_local) == 0:
            return []
        start_local = int(region_local.min()) + max(window, 3)
        limit_local = int(region_local.max())
        candidates = []
        run = 0
        for i in range(start_local, limit_local + 1):
            gi = idx_map[i]
            ok = concave[gi] and (angle_min <= angle_deg[gi] <= angle_max)
            if ok:
                pt = tuple(int(v) for v in chain[i])
                debug_candidates.append((pt, angle_deg[gi], False))
                run += 1
                if run >= min_run:
                    candidates.append((i, pt, int(chain[i][1])))
            else:
                run = 0
        return candidates

    left_candidates = collect_side_candidates(left_chain, left_idx)
    right_candidates = collect_side_candidates(right_chain, right_idx)

    def top_k_near_miss(chain, idx_map, k=3):
        ys_c = chain[:, 1]
        region_local = np.where(ys_c <= y_limit)[0]
        if len(region_local) == 0:
            return []
        scored = []
        for i in region_local:
            gi = idx_map[i]
            if not concave[gi]:
                continue
            a = angle_deg[gi]
            scored.append((abs(a - 90.0), float(a), int(chain[i][1])))
        scored.sort(key=lambda t: t[0])
        return [(round(a, 1), y) for _, a, y in scored[:k]]

    diag = {
        "n_left": len(left_candidates),
        "n_right": len(right_candidates),
        "top3_left": top_k_near_miss(left_chain, left_idx),
        "top3_right": top_k_near_miss(right_chain, right_idx),
    }
    if not left_candidates or not right_candidates:
        return None, None, None, None, left_chain, right_chain, debug_candidates, diag
    best_pair = None
    best_score = None
    for i_l, pt_l, y_l in left_candidates:
        for i_r, pt_r, y_r in right_candidates:
            if abs(pt_l[0] - pt_r[0]) < min_x_sep_frac * width:
                continue
            y_diff = abs(y_l - y_r)
            if y_diff > max_y_dist_frac * height:
                continue
            score = y_diff * 1000 + (y_l + y_r)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (i_l, pt_l, i_r, pt_r)
    if best_pair is None:
        diag["pairing_failed"] = True
        diag["left_ys"] = [y for _, _, y in left_candidates]
        diag["right_ys"] = [y for _, _, y in right_candidates]
        return None, None, None, None, left_chain, right_chain, debug_candidates, diag
    i_left, pt_left, i_right, pt_right = best_pair
    debug_candidates = [
        (pt, a, (pt == pt_left or pt == pt_right)) for pt, a, _ in debug_candidates
    ]
    return (
        pt_left,
        pt_right,
        i_left,
        i_right,
        left_chain,
        right_chain,
        debug_candidates,
        diag,
    )


def run_case1_armpit(chili, placenta_total_px):
    contour_p = get_placenta_contour(chili.mask_placenta)
    if contour_p is None or len(contour_p) < 20:
        return None, None, None, None, contour_p, [], False, "CASE1_NO_CONTOUR"
    pt_l, pt_r, i_left, i_right, left_chain, right_chain, debug_candidates, diag = (
        find_armpit_points(contour_p)
    )
    if pt_l is None or pt_r is None:
        return (
            None,
            None,
            None,
            None,
            contour_p,
            debug_candidates,
            False,
            "CASE1_FAILED",
        )
    s_mask, polygon = build_stem_mask_from_corners(
        chili.mask_placenta.shape, left_chain, i_left, right_chain, i_right
    )
    s_mask = cv2.bitwise_and(s_mask, chili.mask_placenta)
    p_true = cv2.subtract(chili.mask_placenta, s_mask)
    s_px = int((s_mask > 0).sum())
    if s_px >= MIN_STEM_RATIO * placenta_total_px:
        return s_mask, p_true, pt_l, pt_r, contour_p, debug_candidates, True, "CASE1_OK"
    return None, None, None, None, contour_p, debug_candidates, False, "CASE1_TOO_SMALL"


def compute_mask_flesh_raw(chili):
    mask = cv2.subtract(chili.mask_chili, chili.mask_endocarp)
    mask = cv2.subtract(mask, chili.mask_outer)
    mask = cv2.subtract(mask, chili.mask_placenta)
    mask = cv2.subtract(mask, chili.mask_seed)
    return mask


def get_largest_component(mask, min_area=MIN_FLESH_COMPONENT_AREA):
    binary = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return np.zeros_like(binary)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    if areas[largest_label - 1] < min_area:
        return np.zeros_like(binary)
    return (labels == largest_label).astype(np.uint8) * 255


def get_outer_contour(mask_flesh_main):
    contours, _ = cv2.findContours(
        mask_flesh_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return contour


def classify_contour_inner_outer(contour, mask_placenta, dilate_px):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    inner_ref = cv2.dilate(mask_placenta, kernel)
    xs, ys = contour[:, 0], contour[:, 1]
    return inner_ref[ys, xs] > 0


def find_all_true_runs_circular(is_inner, min_run_len=1):
    n = len(is_inner)
    if not is_inner.any() or is_inner.all():
        return []
    extended = np.concatenate([is_inner, is_inner])
    runs = []
    idx = 0
    for val, grp in itertools.groupby(extended):
        length = sum(1 for _ in grp)
        if val and idx < n and length >= min_run_len:
            runs.append((idx % n, (idx + length - 1) % n, length))
        idx += length
    return runs


def get_placenta_top_center(mask_placenta, flesh_center_x):
    ys, xs = np.where(mask_placenta > 0)
    if len(ys) == 0:
        return np.array([flesh_center_x, 0])
    min_y = ys.min()
    top_xs = xs[ys <= min_y + 20]
    return np.array([top_xs.mean() if len(top_xs) > 0 else xs.mean(), min_y])


def refine_tip_from_run(contour, run, ptc, curvature_window=TIP_CURVATURE_WINDOW):
    start, end, _ = run
    n = len(contour)
    idxs = (
        np.arange(start, end + 1)
        if start <= end
        else np.concatenate([np.arange(start, n), np.arange(0, end + 1)])
    )
    pts = contour[idxs]
    if len(idxs) == 1:
        return tuple(int(v) for v in pts[0])
    dist_to_ptc = np.sqrt(np.sum((pts.astype(np.float64) - ptc) ** 2, axis=1))
    max_dist = dist_to_ptc.max()
    dist_norm = dist_to_ptc / max_dist if max_dist > 0 else np.zeros_like(dist_to_ptc)
    w = curvature_window
    curvature_score = np.zeros(len(idxs), dtype=np.float64)
    for i, ci in enumerate(idxs):
        p1 = contour[(ci - w) % n]
        p2 = contour[ci]
        p3 = contour[(ci + w) % n]
        angle, norm_v1, norm_v2 = calculate_angle(p1, p2, p3)
        if norm_v1 == 0 or norm_v2 == 0:
            continue
        curvature_score[i] = (math.pi - angle) / math.pi
    score = TIP_CURVATURE_WEIGHT * curvature_score - TIP_DIST_PENALTY_WEIGHT * dist_norm
    best_idx = int(np.argmax(score))
    return tuple(int(v) for v in pts[best_idx])


def _pick_top_two_tip_points(
    is_inner, contour, mask_flesh_main, mask_placenta, min_run_len
):
    ys, xs = np.where(mask_flesh_main > 0)
    if len(ys) == 0:
        return None, None, {"reason": "empty_flesh_mask"}
    min_y, max_y = ys.min(), ys.max()
    height = max_y - min_y
    flesh_width = max(xs.max() - xs.min(), 1)
    y_threshold = min_y + height * 0.35
    ptc = get_placenta_top_center(mask_placenta, xs.mean())
    runs = find_all_true_runs_circular(is_inner, min_run_len=min_run_len)
    valid_runs = []
    for run in runs:
        tip_pt = refine_tip_from_run(contour, run, ptc)
        if tip_pt[1] < y_threshold:
            valid_runs.append({"pt": tip_pt, "y": tip_pt[1], "x": tip_pt[0]})
    diag = {}
    if len(valid_runs) < 2:
        return None, None, diag
    valid_runs.sort(key=lambda item: item["y"])
    point_a = valid_runs[0]["pt"]
    point_b = None
    max_y_dist = height * 0.15
    for item in valid_runs[1:]:
        x_sep = abs(item["x"] - point_a[0])
        if x_sep >= MIN_TIP_X_SEPARATION_FRAC * flesh_width:
            if abs(item["y"] - point_a[1]) <= max_y_dist:
                point_b = item["pt"]
                break
    return point_a, point_b, diag


def find_arm_tip_points(mask_flesh_main, mask_placenta):
    contour = get_outer_contour(mask_flesh_main)
    if contour is None or len(contour) < 10:
        return None, None, None, None, -1, {"reason": "no_contour_or_too_short"}
    perimeter = len(contour)
    dilate_px = INNER_REF_DILATE_PX_START
    last_diag = {}
    while True:
        is_inner = classify_contour_inner_outer(contour, mask_placenta, dilate_px)
        min_run_len = max(4, int(MIN_INNER_RUN_LEN_FRAC_START * perimeter))
        min_run_len_floor = max(4, int(MIN_INNER_RUN_LEN_FRAC_FLOOR * perimeter))
        for _ in range(RUN_LEN_MAX_ATTEMPTS):
            point_a, point_b, diag = _pick_top_two_tip_points(
                is_inner, contour, mask_flesh_main, mask_placenta, min_run_len
            )
            last_diag = diag
            if point_a and point_b:
                return point_a, point_b, contour, is_inner, dilate_px, diag
            if min_run_len <= min_run_len_floor:
                break
            min_run_len = max(
                min_run_len_floor, int(min_run_len * RUN_LEN_SHRINK_FACTOR)
            )
        if dilate_px >= INNER_REF_DILATE_PX_MAX:
            break
        dilate_px += INNER_REF_DILATE_PX_STEP
    return None, None, contour, None, -1, last_diag


def build_closed_container(mask_flesh_main, point_a, point_b, line_thickness):
    mask_closed = mask_flesh_main.copy()
    cv2.line(mask_closed, point_a, point_b, 255, thickness=line_thickness)
    return ndi.binary_fill_holes(mask_closed > 0).astype(np.uint8) * 255


def run_geometry_box(flesh_mask, placenta_mask):
    p_a, p_b, cnt, is_in, d_used, diag = find_arm_tip_points(flesh_mask, placenta_mask)
    if p_a and p_b:
        if p_a[0] > p_b[0]:
            p_a, p_b = p_b, p_a
        filled_container = build_closed_container(
            flesh_mask, p_a, p_b, CLOSE_LINE_THICKNESS
        )
        p_true = cv2.bitwise_and(placenta_mask, filled_container)
        s_mask = cv2.subtract(placenta_mask, p_true)
        return s_mask, p_true, p_a, p_b, cnt, is_in, d_used, diag
    return None, None, None, None, None, None, -1, diag


def run_case2_geo_original(chili, placenta_total_px):
    mask_flesh_raw = compute_mask_flesh_raw(chili)
    mask_flesh_main = get_largest_component(mask_flesh_raw)
    if mask_flesh_main.sum() == 0:
        return None, None, None, None, None, None, False, "CASE2_FAILED_NO_FLESH"
    s_mask, p_true, p_a, p_b, cnt, is_in, d_used, diag = run_geometry_box(
        mask_flesh_main, chili.mask_placenta
    )
    if s_mask is None:
        return None, None, None, None, None, None, False, "CASE2_FAILED_NO_TIPS"
    return s_mask, p_true, p_a, p_b, cnt, is_in, True, "CASE2_OK"


class Fruits(object):
    def __init__(
        self,
        path,
        model_detect,
        model_segment,
        chili_method=None,
        chili_threshold=None,
        wrinkle_segment_threshold_mm=WRINKLE_SEGMENT_THRESHOLD_MM,
    ):
        self.path = path
        self.model_segment = model_segment
        self.model_detect = model_detect
        self.chili_method = chili_method
        self.chili_threshold = chili_threshold
        if wrinkle_segment_threshold_mm <= 0:
            raise ValueError("wrinkle_segment_threshold_mm must be greater than zero")
        self.wrinkle_segment_threshold_mm = float(wrinkle_segment_threshold_mm)

        qr_result = self.qr_code()
        if qr_result is not None:
            self.ratio, _ = qr_result
            self.scale_source = "qr"
            self.scale_warning = None
        else:
            self.ratio = DEFAULT_DATASET_RATIO_MM_PER_PX
            self.scale_source = "dataset_mean_fallback"
            self.scale_warning = (
                "QR code not found; measurements use the dataset mean scale "
                f"({self.ratio:.9f} mm/pixel)."
            )
            print(f"Warning: {self.scale_warning}")

        self.image = Image.open(path)
        self.width_img, self.height_img = self.image.size

        # Get bounding boxes and sort them from left to right based on X coordinate (bbox[0])
        self.box_cut = self.get_bbox_chili()
        self.box_cut.sort(key=lambda box: box[0])

        self.num_fruit = len(self.box_cut)
        self.items = [self.get_item(i) for i in range(len(self.box_cut))]

    def get_item(self, index):
        chili = Chili(
            self.path,
            index,
            self.model_segment,
            self.box_cut[index],
            self.ratio,
            chili_method=self.chili_method,
            chili_threshold=self.chili_threshold,
            scale_source=self.scale_source,
            wrinkle_segment_threshold_mm=self.wrinkle_segment_threshold_mm,
        )
        print(index, self.box_cut[index])
        return chili

    def show(self):
        img_results = []
        plt.figure(figsize=(20, 10))

        for i in range(len(self.items)):
            img_drawn = draw_full_mask_with_labels(self.items[i].img, self.items[i].polygons)
            img_results.append(img_drawn)

            plt.subplot(1, 6, i + 1)
            plt.imshow(img_drawn)
            plt.title(str(self.items[i].name), fontsize=12)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        # Backward compatibility: the original implementation returned the
        # rendered image list and several notebooks rely on that behavior.
        return img_results

    def get_bbox_chili(self):
        bboxs = self.model_detect.predict(self.path)
        return bboxs

    def qr_code(self):
        path = self.path
        org = korean_imread(path)
        gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        if decode is None:
            return None
        qr_codes = decode(gray)
        if qr_codes:
            qr_code = qr_codes[0]
            x, y, w, h = qr_code.rect
            try:
                SF = 10 / ((w + h) / 2)
                return SF, y + h
            except Exception as e:
                print("Error:", e)
                print(path)
                return None

    def cal_number_seeds(self, bbox, center_seeds):
        center_seeds = np.array(center_seeds)
        if len(center_seeds) == 0:
            return 0
        mask = (
            (center_seeds[:, 0] >= int(bbox[0]))
            & (center_seeds[:, 0] <= int(bbox[2]))
            & (center_seeds[:, 1] >= int(bbox[1]))
            & (center_seeds[:, 1] <= int(bbox[3]))
        )
        points_in_bbox = center_seeds[mask]
        number_seeds = len(points_in_bbox)
        return number_seeds


class Chili(Fruits):
    def __init__(
        self,
        path,
        index,
        model_segment,
        bbox,
        ratio,
        chili_method=None,
        chili_threshold=None,
        scale_source="provided",
        wrinkle_segment_threshold_mm=WRINKLE_SEGMENT_THRESHOLD_MM,
    ):
        self.path = path
        self.bbox = bbox
        self.model_segment = model_segment
        self.id = index
        self.ratio = ratio
        self.scale_source = scale_source
        if wrinkle_segment_threshold_mm <= 0:
            raise ValueError("wrinkle_segment_threshold_mm must be greater than zero")
        self.wrinkle_segment_threshold_mm = float(wrinkle_segment_threshold_mm)
        self._wall_feature_data = None
        self.name = self.get_name()
        self.position, self.img = self.get_img(self.bbox)

        self.width_bbox, self.height_bbox = self.get_width_height_bbox()

        self.rseg = model_segment.predict(
            self.img,
            chili_method=chili_method,
            chili_threshold=chili_threshold,
        )
        self.polygons = self.get_polygons()

        self.mask_chili = self._get_class_mask(4)
        # Endocarp is consumed directly from YOLO's raster masks.  Do not use
        # SegmentModel's filled/smoothed polygon representation for this class.
        self.mask_endocarp = self._get_raw_yolo_class_mask(3)
        self.mask_endocarp_source = "raw_yolo"
        self.mask_placenta = self._get_class_mask(2)
        self.mask_outer = self._get_class_mask(1)
        self.mask_seed = self._get_class_mask(0)

        combined = cv2.bitwise_or(self.mask_endocarp, self.mask_placenta)
        combined = cv2.bitwise_or(combined, self.mask_outer)
        combined = cv2.bitwise_or(combined, self.mask_seed)
        self.mask_chili = cv2.bitwise_or(self.mask_chili, combined)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.mask_chili = cv2.morphologyEx(self.mask_chili, cv2.MORPH_CLOSE, kernel_close)
        self.mask_chili = cv2.morphologyEx(self.mask_chili, cv2.MORPH_OPEN, kernel_open)

        
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(
                (self.mask_chili > 0).astype(np.uint8),
                connectivity=8,
            )
        )
        total_area = int(np.count_nonzero(self.mask_chili))
        minimum_area = max(
            CHILI_MIN_COMPONENT_AREA,
            int(round(total_area * CHILI_MIN_COMPONENT_RATIO)),
        )
        filtered_mask = np.zeros_like(self.mask_chili)
        for component_id in range(1, component_count):
            area = component_stats[component_id, cv2.CC_STAT_AREA]
            if area >= minimum_area:
                filtered_mask[component_labels == component_id] = 255
        if np.any(filtered_mask):
            self.mask_chili = filtered_mask

        self.mask_chili = (
            ndi.binary_fill_holes(self.mask_chili > 0).astype(np.uint8) * 255
        )

        # Update all class 4 polygons so display functions use the same
        contours, _ = cv2.findContours(self.mask_chili, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.polygons = [p for p in self.polygons if p['class_id'] != 4]
        for contour in contours:
            if cv2.contourArea(contour) <= 0:
                continue
            pts = contour.reshape(-1, 2).astype(np.float32)
            self.polygons.append({
                "class_id": 4,
                "class_name": "chili",
                "points": pts
            })

        self.mask = self.mask_chili

        # STEM SEPARATION
        self.mask_stem, self.mask_true_placenta = self._compute_stem_and_true_placenta()


        self.spine = self._calculate_spine()

        self.width_fruit_stats = self.get_width_fruit()
        self.length_fruit = self.get_length_fruit()
        self.width_fruit_9seg, self.length_fruit_9seg = self.get_width_length_fruit_9seg()

        # Keep the original public scalar name. The enhanced implementation
        # additionally exposes min/max/avg through width_fruit_stats.
        self.width_fruit = self.width_fruit_stats["avg"]

        # Keep the original area name and semantics: area of the complete
        # fruit mask, scaled by the QR-derived ratio.
        self.area_fruit = self.get_area()

        self.endocarp_area = self.get_endocarp_area()
        self.fruit_body_area = self.get_fruit_body_area()
        self.placenta_area = self.get_placenta_area()
        self.thickness_stats = self.get_thickness()

        red_metrics = self.get_redness_metrics()
        self.redness = red_metrics.get("redness_chroma", 0)
        self.color_uniformity = red_metrics.get("color_uniformity", 0)
        self.wrinkle = self.get_wrinkle()

        self.flesh_placenta_ratio = self.get_flesh_placenta_ratio()
        self.seed_stats = self.get_seed_stats()
        self.number_seeds = self.seed_stats["count"]

        # 3 Advanced Features
        self.curvature_index = self.get_curvature_index()
        self.seed_density = self.get_seed_density()
        self.asymmetry_index = self.get_asymmetry_index()

    def _compute_stem_and_true_placenta(self):
        mask_placenta = self.mask_placenta
        if mask_placenta.sum() == 0:
            return np.zeros_like(mask_placenta), mask_placenta.copy()
        placenta_total_px = int((mask_placenta > 0).sum())
        mask_stem = np.zeros_like(mask_placenta)
        mask_placenta_true = mask_placenta.copy()
        s_mask1, p_true1, pa1, pb1, contour_p, debug_candidates, case1_ok, log1 = (
            run_case1_armpit(self, placenta_total_px)
        )
        if case1_ok:
            return s_mask1, p_true1
        s_mask2, p_true2, pa2, pb2, cnt2, is_in2, case2_ok, log2 = (
            run_case2_geo_original(self, placenta_total_px)
        )
        if case2_ok:
            return s_mask2, p_true2
        return mask_stem, mask_placenta_true

    def get_img(self, bbox):
        img = cv2.imread(self.path)
        if img is None:
            return 0
        H, W, _ = img.shape
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = bbox
        pad = 50
        x1, x2, y1, y2 = (
            max(0, x - pad),
            min(W, x + w + pad),
            max(0, y - pad),
            min(H, y + h + pad),
        )
        fruit = img_RGB[y1:y2, x1:x2]
        return [x1, x2, y1, y2], fruit

    def get_width_height_bbox(self):
        img = self.img
        bbox = self.bbox
        ratio = self.ratio
        self.rotation = "none"
        #Bbox has format [x, y, w, h]
        width_bbox = round(bbox[2] * ratio, 2)
        height_bbox = round(bbox[3] * ratio, 2)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        low_green = np.array([25, 20, 20])
        high_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv_img, low_green, high_green)

        centroid_pedicel = find_centroid(green_mask)

        if centroid_pedicel is not None:
            cx, cy = centroid_pedicel
            crop_height, crop_width = img.shape[:2]

            # Normalize each distance by its corresponding image dimension.
            # Raw pixel distances are not comparable for a rectangular crop:
            # vertical distances belong to height, horizontal distances to width.
            edge_distances = {
                "top": cy / crop_height,
                "bottom": (crop_height - cy) / crop_height,
                "left": cx / crop_width,
                "right": (crop_width - cx) / crop_width,
            }
            pedicel_edge = min(edge_distances, key=edge_distances.get)

            if pedicel_edge == "bottom":
                img = cv2.rotate(img, cv2.ROTATE_180)
                self.rotation = "180"
            elif pedicel_edge == "left":
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                self.rotation = "90_cw"
                width_bbox, height_bbox = height_bbox, width_bbox
            elif pedicel_edge == "right":
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.rotation = "90_ccw"
                width_bbox, height_bbox = height_bbox, width_bbox

            self.img = img

        return width_bbox, height_bbox

    def get_polygons(self):
        return self.rseg["polygons"]

    def _get_class_mask(self, class_id):
        polygons = self.rseg.get("polygons", [])
        H, W = self.img.shape[:2]
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        if polygons:
            for poly_data in polygons:
                if poly_data["class_id"] == class_id:
                    pts = poly_data["points"].astype(np.int32)
                    cv2.fillPoly(combined_mask, [pts], 255)
        else:
            masks_data = self.rseg.get("masks_data", [])
            for md in masks_data:
                if md["class_id"] == class_id:
                    single_mask = (md["mask"] > 0).astype(np.uint8) * 255
                    combined_mask = cv2.bitwise_or(combined_mask, single_mask)
        return combined_mask

    def _get_raw_yolo_class_mask(self, class_id):
        """Combine raw YOLO raster masks without polygon or hole postprocessing."""
        height, width = self.img.shape[:2]
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for mask_data in self.rseg.get("masks_data", []):
            if int(mask_data.get("class_id", -1)) != int(class_id):
                continue
            if mask_data.get("source", "yolo") != "yolo":
                continue

            single_mask = (mask_data["mask"] > 0).astype(np.uint8) * 255
            if single_mask.shape != (height, width):
                single_mask = cv2.resize(
                    single_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            combined_mask = cv2.bitwise_or(combined_mask, single_mask)
        return combined_mask

    def get_name(self):
        path = Path(self.path)
        return f"{path.stem}_{self.id}.jpg"

    def _calculate_spine(self):
        mask = cv2.subtract(self.mask_chili, self.mask_outer)
        nonzero_y, nonzero_x = np.where(mask > 0)
        if len(nonzero_y) == 0:
            return None
        y_min, y_max = np.min(nonzero_y), np.max(nonzero_y)
        y_vals = np.arange(y_min, y_max + 1)
        mid_x = []
        for y in y_vals:
            row = mask[y]
            xs = np.where(row > 0)[0]
            if len(xs) == 0:
                mid_x.append(np.nan)
            else:
                mid_x.append((xs[0] + xs[-1]) / 2.0)
        mid_x = np.array(mid_x)
        valid = ~np.isnan(mid_x)
        if valid.sum() < 4:
            return None
        coeffs = np.polyfit(y_vals[valid], mid_x[valid], 3)
        return {
            "coeffs": coeffs,
            "y_min": y_min,
            "y_max": y_max,
            "y_vals": y_vals,
            "valid": valid,
        }

    def get_width_fruit(self):
        mask = cv2.subtract(self.mask_chili, self.mask_outer)
        nonzero_y, nonzero_x = np.where(mask > 0)
        if len(nonzero_y) == 0:
            return {"min": 0, "max": 0, "avg": 0}
        y_min, y_max = np.min(nonzero_y), np.max(nonzero_y)
        h = y_max - y_min
        valid_y_min = y_min + int(0.05 * h)
        valid_y_max = y_max - int(0.05 * h)
        widths = []
        for i in range(valid_y_min, valid_y_max + 1):
            if i < 0 or i >= mask.shape[0]:
                continue
            row = mask[i]
            nonzero_indices = np.where(row > 0)[0]
            if len(nonzero_indices) > 0:
                widths.append(nonzero_indices[-1] - nonzero_indices[0])
        if not widths:
            return {"min": 0, "max": 0, "avg": 0}
        return {
            "min": round(min(widths) * self.ratio, 2),
            "max": round(max(widths) * self.ratio, 2),
            "avg": round(np.mean(widths) * self.ratio, 2),
        }

    def get_length_fruit(self):
        if not hasattr(self, "spine") or self.spine is None:
            return 0
        coeffs = self.spine["coeffs"]
        y_vals = self.spine["y_vals"]
        valid = self.spine["valid"]
        poly_deriv = np.polyder(np.poly1d(coeffs))
        dx_dy = poly_deriv(y_vals[valid])
        length_px = np.sum(np.sqrt(1 + dx_dy**2))
        return round(length_px * self.ratio, 2)

    def get_width_length_fruit_9seg(self):
        mask = self.mask_chili
        top = 0
        bottom = 0
        for i in range(len(mask)):
            temp = mask[i]
            if temp.sum() > 0:
                if top == 0:
                    top = i
                bottom = i
        d_pin = int((bottom - top) / 9)
        pins = []
        for i in range(9):
            pins.append(top + i * d_pin)
        pins.append(bottom)
        points = []
        length_fruit = 0
        width_fruit = 0
        for i in pins:
            temp = mask[i]
            max_j = -1
            min_j = -1
            for j in range(len(temp)):
                if temp[j] > 0:
                    if min_j == -1:
                        min_j = j
                    max_j = j
            point = [i, min_j, max_j]
            if i > top and i < bottom:
                width_fruit += max_j - min_j
            points.append(point)
        width_fruit /= 8
        for i in range(len(points)):
            if i == 0:
                continue
            p1 = [points[i - 1][0], (points[i - 1][1] + points[i - 1][2]) / 2]
            p2 = [points[i][0], (points[i][1] + points[i][2]) / 2]
            length_fruit += math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        ratio = self.ratio
        width_fruit = round(width_fruit * ratio, 2)
        length_fruit = round(length_fruit * ratio, 2)
        return width_fruit, length_fruit

    def get_endocarp_area(self):
        mask_flesh = cv2.subtract(self.mask_chili, self.mask_endocarp)
        mask_flesh = cv2.subtract(mask_flesh, self.mask_outer)
        mask_flesh = cv2.subtract(mask_flesh, self.mask_stem)
        mask_flesh = cv2.subtract(mask_flesh, self.mask_true_placenta)
        mask_flesh = cv2.subtract(mask_flesh, self.mask_seed)
        return round((mask_flesh / 255).sum() * (self.ratio**2), 2)

    def get_thickness(self):
        return calculate_thickness_stats(self._get_wall_feature_data(), self.ratio)

    def _get_wall_feature_data(self):
        """Build wall geometry once and share it between thickness and wrinkle."""
        if self._wall_feature_data is None:
            self._wall_feature_data = build_wall_data(self, self.ratio)
        return self._wall_feature_data

    def get_fruit_body_area(self):
        mask = cv2.subtract(self.mask_chili, self.mask_outer)
        mask = cv2.subtract(mask, self.mask_stem)
        return round((mask / 255).sum() * (self.ratio**2), 2)

    def get_placenta_area(self):
        mask = cv2.subtract(self.mask_true_placenta, self.mask_outer)
        mask = cv2.subtract(mask, self.mask_seed)
        return round((mask / 255).sum() * (self.ratio**2), 2)

    def get_flesh_placenta_ratio(self):
        flesh_area = self.get_endocarp_area()
        placenta_area = self.get_placenta_area()
        if placenta_area == 0:
            return None
        return round(flesh_area / placenta_area, 2)

    MIN_SEED_PEAK_DISTANCE_PX = 5

    def _estimate_min_distance(self, reference_area_px):
        est_radius_px = np.sqrt(max(reference_area_px, 0) / np.pi)
        return max(3, int(est_radius_px * 0.8))

    def get_seed_stats(self):
        binary_mask = self.mask_seed > 0
        if binary_mask.sum() == 0:
            return {"sum": 0, "min": 0, "max": 0, "avg": 0, "std": 0, "count": 0}
        seed_masks = [
            (md["mask"] > 0)
            for md in self.rseg.get("masks_data", [])
            if md.get("class_id") == 0
        ]
        all_instance_areas = [int(m.sum()) for m in seed_masks]
        global_median_area_px = (
            float(np.median(all_instance_areas)) if all_instance_areas else 0.0
        )
        cluster_labels, num_clusters = ndi.label(binary_mask)
        dist_transform = ndi.distance_transform_edt(binary_mask)
        coords = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = cluster_labels == cluster_id
            cluster_area_px = int(cluster_mask.sum())
            overlapping_areas = [
                int(m.sum()) for m in seed_masks if np.any(m & cluster_mask)
            ]
            if overlapping_areas:
                reference_area_px = float(np.median(overlapping_areas))
            elif global_median_area_px > 0:
                reference_area_px = global_median_area_px
            else:
                reference_area_px = float(cluster_area_px)
            expected_count = 1
            if reference_area_px > 0:
                expected_count = max(1, int(round(cluster_area_px / reference_area_px)))
            if expected_count <= 1:
                cluster_dist = np.where(cluster_mask, dist_transform, -1)
                y, x = np.unravel_index(np.argmax(cluster_dist), cluster_dist.shape)
                coords.append((y, x))
                continue
            cluster_dist_only = np.where(cluster_mask, dist_transform, 0)
            min_distance_px = self._estimate_min_distance(reference_area_px)
            local_coords = peak_local_max(
                cluster_dist_only,
                min_distance=min_distance_px,
                exclude_border=False,
                num_peaks=expected_count,
            )
            attempts = 0
            while (
                len(local_coords) < expected_count
                and min_distance_px > 3
                and attempts < 6
            ):
                min_distance_px = max(3, int(min_distance_px * 0.75))
                local_coords = peak_local_max(
                    cluster_dist_only,
                    min_distance=min_distance_px,
                    exclude_border=False,
                    num_peaks=expected_count,
                )
                attempts += 1
            if len(local_coords) == 0:
                y, x = np.unravel_index(
                    np.argmax(cluster_dist_only), cluster_dist_only.shape
                )
                coords.append((y, x))
            else:
                coords.extend([tuple(c) for c in local_coords])
        markers = np.zeros(dist_transform.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i
        labels_ws = sk_watershed(-dist_transform, markers, mask=binary_mask)
        areas = []
        for label_id in range(1, int(labels_ws.max()) + 1):
            area_px = int((labels_ws == label_id).sum())
            if area_px >= 5:
                areas.append(area_px * (self.ratio**2))
        if not areas:
            return {"sum": 0, "min": 0, "max": 0, "avg": 0, "std": 0, "count": 0}
        return {
            "sum": round(sum(areas), 2),
            "min": round(min(areas), 2),
            "max": round(max(areas), 2),
            "avg": round(np.mean(areas), 2),
            "std": round(np.std(areas), 2),
            "count": len(areas),
        }

    def get_redness(self):
        #Return a scalar under the original public method name.
        return self.get_redness_metrics()["redness_chroma"]


    def get_curvature_index(self):
        if not hasattr(self, "spine") or self.spine is None:
            return 0
        if self.length_fruit == 0:
            return 0
        coeffs = self.spine["coeffs"]
        y_min, y_max = self.spine["y_min"], self.spine["y_max"]
        poly = np.poly1d(coeffs)
        x_top = poly(y_min)
        x_bottom = poly(y_max)
        straight_dist_px = math.sqrt((x_top - x_bottom) ** 2 + (y_min - y_max) ** 2)
        straight_dist_mm = straight_dist_px * self.ratio
        curvature = straight_dist_mm / self.length_fruit
        return round(float(np.clip(curvature, 0.0, 1.0)), 4)

    def get_seed_density(self):
        seed_px = (self.mask_seed > 0).sum()
        cavity_mask = cv2.bitwise_or(self.mask_endocarp, self.mask_seed)
        cavity_mask = cv2.bitwise_or(cavity_mask, self.mask_true_placenta)
        cavity_px = (cavity_mask > 0).sum()
        if cavity_px == 0:
            return 0
        density = seed_px / cavity_px
        return round(float(density), 4)

    def get_asymmetry_index(self):
        if not hasattr(self, "spine") or self.spine is None:
            return 0
        mask_body = cv2.subtract(self.mask_chili, self.mask_outer)
        y_vals = self.spine["y_vals"]
        valid = self.spine["valid"]
        poly = np.poly1d(self.spine["coeffs"])
        sum_diffs = 0
        total_area = 0
        for y, is_valid in zip(y_vals, valid):
            if not is_valid:
                continue
            row = mask_body[y]
            xs = np.where(row > 0)[0]
            if len(xs) == 0:
                continue
            x_spine = poly(y)
            x_left = xs[0]
            x_right = xs[-1]
            x_spine_clamped = max(x_left, min(x_right, x_spine))
            width_left = x_spine_clamped - x_left
            width_right = x_right - x_spine_clamped
            sum_diffs += abs(width_left - width_right)
            total_area += width_left + width_right
        if total_area == 0:
            return 0
        asymmetry = sum_diffs / total_area
        return round(float(asymmetry), 4)

    def segm(self, threshold=56):
        #Preserve the original threshold helper for external callers.
        img_red = np.asarray(self.img[:, :, 0])
        return np.where(img_red >= threshold, 255, 0)

    def get_mask(self, use_closing=True, use_contour=True):
        return self.mask_chili.copy()

    def get_width_length_fruit(self):
        #Return width and length using the original public method name.
        return self.width_fruit_stats["avg"], self.length_fruit

    def get_area(self):
        #Calculate complete fruit area using the original public name.
        return round((self.mask_chili > 0).sum() * (self.ratio ** 2), 2)

    def get_wrinkle(self, wrinkle_segment_threshold_mm=None):
        """Calculate mean curvature excess on the true outer wall edge."""
        threshold = (
            self.wrinkle_segment_threshold_mm
            if wrinkle_segment_threshold_mm is None
            else float(wrinkle_segment_threshold_mm)
        )
        return calculate_wrinkle(self._get_wall_feature_data(), self.ratio, threshold)

    def get_hist(self):
        #Preserve the original red-channel histogram visualization helper.
        img_red = np.asarray(self.img[:, :, 0])
        hist = cv2.calcHist([img_red], [0], None, [256], [0, 256])
        height, width = 400, 800
        hist_image = np.zeros((height, width, 3), dtype=np.uint8)
        hist = cv2.normalize(
            hist, hist, alpha=0, beta=height, norm_type=cv2.NORM_MINMAX
        )
        for i in range(256):
            cv2.line(
                hist_image,
                (i, height),
                (i, int(height - hist[i, 0])),
                (255, 255, 255),
                1,
            )
        return hist_image

    def get_contour(self, ep=0.005):
        #Preserve the original contour inspection helper.
        img = self.img.copy()
        contours, _ = cv2.findContours(
            self.mask_chili, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return contours, img

        cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
        return contours, img

    def get_redness_metrics(self):
        mask_remove = cv2.bitwise_or(self.mask_seed, self.mask_outer)
        mask_remove = cv2.bitwise_or(mask_remove, self.mask_placenta)
        mask_pure_red = cv2.subtract(self.mask_chili, mask_remove)

        lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)

        #In OpenCV, a* is scaled to [0, 255] with 128 being the zero point
        a_channel = lab[:, :, 1].astype(np.float32)
        ripeness_a_star = a_channel - 128.0

        pure_red_pixels = ripeness_a_star[mask_pure_red > 0]
        if len(pure_red_pixels) == 0:
            return {"redness_chroma": 0, "color_uniformity": 0}

        return {
            "redness_chroma": round(float(np.mean(pure_red_pixels)), 4),
            "color_uniformity": round(float(np.std(pure_red_pixels)), 4)
        }
