from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_INPUT_SIZE = 640
DEFAULT_CONFIDENCE = 0.70
DEFAULT_IOU = 0.30
DEFAULT_CONTAINMENT = 0.60
DEFAULT_CROP_TOP = 540
CHILI_CLASS_ID = 1
DEFAULT_ONNX_WEIGHT = (
    Path(__file__).resolve().parents[1] / "weights" / "yolov7_detection.onnx"
)


EP = 0.003


def letterbox(image, new_shape=DEFAULT_INPUT_SIZE, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio and symmetric padding."""
    if image is None or image.size == 0:
        raise ValueError("Cannot letterbox an empty image.")
    if isinstance(new_shape, int):
        target_height = target_width = new_shape
    else:
        target_height, target_width = new_shape

    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if (resized_width, resized_height) != (width, height):
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
    else:
        resized = image.copy()

    padding_width = target_width - resized_width
    padding_height = target_height - resized_height
    left = padding_width // 2
    right = padding_width - left
    top = padding_height // 2
    bottom = padding_height - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (padding_width / 2.0, padding_height / 2.0)


def box_iou_one_to_many(box, boxes):
    """Calculate IoU between one xyxy box and an array of xyxy boxes."""
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    intersection = (
        np.maximum(0.0, intersection_x2 - intersection_x1)
        * np.maximum(0.0, intersection_y2 - intersection_y1)
    )
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = box_area + boxes_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def box_intersection_over_min_area(box, boxes):
    """Return intersection divided by the smaller box area (IoM)."""
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    intersection = (
        np.maximum(0.0, intersection_x2 - intersection_x1)
        * np.maximum(0.0, intersection_y2 - intersection_y1)
    )
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    smaller_area = np.minimum(box_area, boxes_area)
    return np.divide(
        intersection,
        smaller_area,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=smaller_area > 0,
    )


def non_max_suppression(
    boxes,
    scores,
    iou_threshold,
    containment_threshold=DEFAULT_CONTAINMENT,
):
    """Return indices kept by class-specific IoU/containment NMS."""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = box_iou_one_to_many(boxes[current], boxes[remaining])
        containments = box_intersection_over_min_area(
            boxes[current], boxes[remaining]
        )
        order = remaining[
            (overlaps < iou_threshold)
            & (containments < containment_threshold)
        ]
    return keep


def restore_boxes(boxes, scale, padding, source_shape, crop_top=0):
    """Undo letterbox coordinates and map ROI boxes onto the source image."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    source_height, source_width = source_shape[:2]
    pad_x, pad_y = padding
    restored = np.asarray(boxes, dtype=np.float32).copy()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - pad_x) / scale
    restored[:, [1, 3]] = (restored[:, [1, 3]] - pad_y) / scale
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, source_width)
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, source_height)
    restored[:, [1, 3]] += crop_top
    return restored


def decode_yolov7_output(
    output,
    confidence_threshold=DEFAULT_CONFIDENCE,
    iou_threshold=DEFAULT_IOU,
    chili_class_id=CHILI_CLASS_ID,
    containment_threshold=DEFAULT_CONTAINMENT,
):
    """Decode raw YOLOv7 [cx,cy,w,h,obj,class...] output into xyxy boxes."""
    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[1] < 7:
        raise ValueError(
            "YOLOv7 output must have shape [1, N, 7] or [N, 7]."
        )
    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = predictions[:, 4] * class_scores[
        np.arange(len(predictions)), class_ids
    ]
    selected = (class_ids == chili_class_id) & (scores >= confidence_threshold)
    selected_predictions = predictions[selected]
    selected_scores = scores[selected].astype(np.float32)
    if len(selected_predictions) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    xywh = selected_predictions[:, :4].astype(np.float32)
    boxes = np.column_stack(
        (
            xywh[:, 0] - xywh[:, 2] / 2,
            xywh[:, 1] - xywh[:, 3] / 2,
            xywh[:, 0] + xywh[:, 2] / 2,
            xywh[:, 1] + xywh[:, 3] / 2,
        )
    )
    kept = non_max_suppression(
        boxes,
        selected_scores,
        iou_threshold,
        containment_threshold,
    )
    return boxes[kept], selected_scores[kept]


def hadle_segm(img, threshold=56):
    # print(type(img))
    img = img.copy()
    img = np.array(img)
    img = np.where(img >= threshold, 255, 0)

    return img
    
def get_mask(img,threshold = 16, use_closing = True, use_contour = True):
    img = img
    
    img_red = np.array(img[:, :, 0])
    hist = cv2.calcHist([img_red], [0], None, [256], [0, 256])

    fruit_mask = hadle_segm(img_red, hist.argmax() + threshold)
    fruit_mask = fruit_mask.astype(np.uint8)

    if use_closing:
        kernel = np.ones((10, 10), np.uint8)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        fruit_mask = fruit_mask.astype(np.uint8)
    if use_contour:
        contours, hierarchy = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(fruit_mask)
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        max_area = max(contour_areas)
        fruit_mask = np.zeros_like(fruit_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= max_area:
                cv2.drawContours(fruit_mask, [cnt], -1, (255, 255, 255), -1)

    return fruit_mask

def check_object(img):
    mask = get_mask(img, use_closing=False, use_contour=False)

    if mask.sum() > 0:
        return True
    else:
        return False


class DetectModel:
    def __init__(
        self,
        model_name=None,
        weight_path=None,
        device="cpu",
        *,
        backend=None,
        conf_threshold=DEFAULT_CONFIDENCE,
        iou_threshold=DEFAULT_IOU,
        containment_threshold=DEFAULT_CONTAINMENT,
        input_size=DEFAULT_INPUT_SIZE,
        crop_top=DEFAULT_CROP_TOP,
    ):
        del device  # YOLOv7 ONNX detection intentionally runs on CPU.
        if backend is None:
            legacy_backends = {
                None: "yolov7_onnx",
                "none": "threshold",
                "yolo": "yolov7_onnx",
                "yolov7_onnx": "yolov7_onnx",
                "threshold": "threshold",
            }
            if model_name not in legacy_backends:
                raise ValueError(f"Unsupported detector model_name: {model_name}")
            backend = legacy_backends[model_name]
        if backend not in {"yolov7_onnx", "threshold"}:
            raise ValueError(
                "Unsupported detector backend. Use 'yolov7_onnx' or 'threshold'."
            )

        self.backend = backend
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.containment_threshold = float(containment_threshold)
        self.input_size = int(input_size)
        self.crop_top = int(crop_top)
        if not 0 <= self.conf_threshold <= 1:
            raise ValueError("conf_threshold must be between 0 and 1.")
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if not 0 <= self.containment_threshold <= 1:
            raise ValueError("containment_threshold must be between 0 and 1.")
        if self.input_size <= 0:
            raise ValueError("input_size must be greater than zero.")
        if self.crop_top < 0:
            raise ValueError("crop_top cannot be negative.")
        self.weight_path = Path(weight_path or DEFAULT_ONNX_WEIGHT)
        self.session = None
        self.input_name = None
        self.output_name = None

        if backend == "yolov7_onnx":
            self._load_onnx_session()

    def _load_onnx_session(self):
        if not self.weight_path.is_file():
            raise FileNotFoundError(
                f"YOLOv7 ONNX weight not found: {self.weight_path}"
            )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.weight_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        active_providers = self.session.get_providers()
        if not active_providers or active_providers[0] != "CPUExecutionProvider":
            raise RuntimeError(
                f"YOLOv7 detector requires CPUExecutionProvider; got {active_providers}."
            )
        self._validate_onnx_signature()

    def _validate_onnx_signature(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        expected_input_shape = [1, 3, self.input_size, self.input_size]
        if len(inputs) != 1 or list(inputs[0].shape) != expected_input_shape:
            actual = [item.shape for item in inputs]
            raise ValueError(
                f"YOLOv7 ONNX input must be {expected_input_shape}; got {actual}."
            )
        if inputs[0].type != "tensor(float)":
            raise ValueError(
                f"YOLOv7 ONNX input must be tensor(float); got {inputs[0].type}."
            )
        if len(outputs) != 1 or len(outputs[0].shape) != 3:
            actual = [item.shape for item in outputs]
            raise ValueError(
                f"YOLOv7 ONNX must have one [1,N,7] output; got {actual}."
            )
        output_shape = list(outputs[0].shape)
        if output_shape[0] != 1 or output_shape[-1] != 7:
            raise ValueError(
                f"YOLOv7 ONNX output must end in 7 values; got {output_shape}."
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def model_detect(self, image_rgb):
        prepared, scale, padding = letterbox(image_rgb, self.input_size)
        tensor = np.ascontiguousarray(
            prepared.transpose(2, 0, 1)[None], dtype=np.float32
        )
        tensor /= 255.0
        output = self.session.run(
            [self.output_name], {self.input_name: tensor}
        )[0]
        boxes, scores = decode_yolov7_output(
            output,
            confidence_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            containment_threshold=self.containment_threshold,
        )
        return boxes, scores, scale, padding

    def handle_detect(self, image, offset_y=DEFAULT_CROP_TOP):
        ep = EP
        img_RGB = image.copy()
        img_mask = get_mask(img_RGB, 22 , True, False)
        
        contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fruit_mask = img_RGB
        bboxs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 10000:
                cv2.drawContours(fruit_mask, [cnt], -1, (255, 255, 255), 2)
                length_L1 = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, ep*length_L1, True)
                contour = [p[0].tolist() for p in cnt]
                contour_or = [[p[0], p[1] + offset_y] for p in contour]
                x, y, w, h = cv2.boundingRect(np.array(contour_or))
                bboxs.append([x, y, w, h])
                
        return fruit_mask, bboxs

    def predict(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"OpenCV cannot read input image: {path}")
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_RGB.shape[:2]
        if image_height <= self.crop_top:
            return []
        image_crop = image_RGB[self.crop_top:, :, :]

        if self.backend == "yolov7_onnx":
            boxes, _scores, scale, padding = self.model_detect(image_crop)
            boxes = restore_boxes(
                boxes,
                scale,
                padding,
                image_crop.shape,
                crop_top=self.crop_top,
            )
            bboxs = []
            for x1, y1, x2, y2 in boxes:
                x1 = int(np.rint(np.clip(x1, 0, image_width)))
                y1 = int(np.rint(np.clip(y1, 0, image_height)))
                x2 = int(np.rint(np.clip(x2, 0, image_width)))
                y2 = int(np.rint(np.clip(y2, 0, image_height)))
                if x2 > x1 and y2 > y1:
                    bboxs.append([x1, y1, x2 - x1, y2 - y1])
        else:
            _, bboxs = self.handle_detect(
                image_crop, offset_y=self.crop_top
            )
        
        bboxs = self.sort_bboxes_by_variance(bboxs)
        return bboxs
    
    def sort_bboxes_by_variance(self, bboxs):
        bboxs = list(bboxs)
        if not bboxs:
            return []
        x_coords = [bbox[0] for bbox in bboxs]
        y_coords = [bbox[1] for bbox in bboxs]
        var_x = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 0
        var_y = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 0
        
        if var_x > var_y:
            bboxs.sort(key=lambda b: b[0])
        else:
            bboxs.sort(key=lambda b: b[1])
            
        return bboxs