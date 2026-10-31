"""Shared runtime configuration, exports and provenance for Web and CLI.

Measurement algorithms live in object/geometric_features and are unchanged.
Diagnostics are deliberately separate from the stable numeric CSV schema.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
import json
import math
from pathlib import Path
import subprocess
import platform

from src.traits import CANONICAL_TRAIT_COLUMNS, canonical_trait_row

BASE_DIR = Path(__file__).resolve().parents[1]
EXTRA_COLUMNS = (
    "Width BBox", "Height BBox", "Width (9 segments)", "Length (9 segments)",
    "Fruit Body Area", "Seed Area Min", "Seed Area Max", "Seed Area Average",
    "Seed Area Std", "Redness", "Color Uniformity", "Scale Ratio", "Scale Source",
)


@dataclass(frozen=True)
class PipelineConfig:
    detector_backend: str = "yolov7_onnx"
    weight_det: str = str(BASE_DIR / "weights/yolov7_detection.onnx")
    weight_seg: str = str(BASE_DIR / "weights/yolov11m_segment_final.pt")
    device: str = "cpu"
    det_conf: float = 0.70
    det_iou: float = 0.30
    det_containment: float = 0.60
    det_input_size: int = 640
    det_crop_top: int = 540
    seg_imgsz: int = 640
    fruit_threshold_offset: int = 16
    wrinkle_tau_mm: float = 5.0

    def __post_init__(self):
        if self.detector_backend not in {"yolov7_onnx", "threshold"}:
            raise ValueError("Unsupported detector_backend")
        for name in ("det_conf", "det_iou", "det_containment"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name in ("det_input_size", "seg_imgsz", "det_crop_top", "fruit_threshold_offset"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.det_input_size <= 0 or self.seg_imgsz <= 0 or self.det_crop_top < 0:
            raise ValueError("Input sizes must be positive; crop_top must be nonnegative")
        if not 0 <= self.fruit_threshold_offset <= 255:
            raise ValueError("fruit_threshold_offset must be between 0 and 255")
        if not math.isfinite(self.wrinkle_tau_mm) or self.wrinkle_tau_mm <= 0:
            raise ValueError("wrinkle_tau_mm must be finite and positive")

    def effective_settings(self):
        result = asdict(self)
        result.update({"detector_device": "cpu", "detector_threshold_offset": 22,
                       "detector_min_contour_area_px": 10000,
                       "crop_padding_px": 50, "seg_iou": 0.7,
                       "seg_seed_conf": 0.35, "seg_component_conf": 0.10,
                       "seg_model_conf": 0.01, "retina_masks": True,
                       "fruit_mask_source": "red_channel_threshold"})
        return result


def build_models(config):
    from src.detect import DetectModel
    from src.segment import SegmentModel
    return (
        DetectModel(backend=config.detector_backend, weight_path=config.weight_det,
                    conf_threshold=config.det_conf, iou_threshold=config.det_iou,
                    containment_threshold=config.det_containment,
                    input_size=config.det_input_size, crop_top=config.det_crop_top),
        SegmentModel(weight_path=config.weight_seg, device=config.device,
                     imgsz=config.seg_imgsz,
                     fruit_threshold_offset=config.fruit_threshold_offset),
    )


def run_image(path, config, models=None):
    import cv2
    from src.object import Fruits
    path = Path(path)
    if not path.is_file() or cv2.imread(str(path)) is None:
        raise ValueError(f"Unreadable input image: {path}")
    detector, segmenter = models if models is not None else build_models(config)
    # Injected models must agree with the settings recorded in the manifest.
    for attr, expected in (("imgsz", config.seg_imgsz), ("device", config.device)):
        if str(getattr(segmenter, attr)) != str(expected):
            raise ValueError(f"Segment model {attr} differs from PipelineConfig")
    return Fruits(str(path), model_detect=detector, model_segment=segmenter,
                  fruit_threshold_offset=config.fruit_threshold_offset,
                  mesocarp_wrinkle_segment_threshold_mm=config.wrinkle_tau_mm)


def feature_columns(include_root=False):
    return (["ID", "Name Root", "Name"] if include_root else ["ID", "Name"]) + list(CANONICAL_TRAIT_COLUMNS) + list(EXTRA_COLUMNS)


def feature_row(chili, root_name=None):
    row = {"ID": chili.id, "Name": chili.name}
    if root_name is not None:
        row["Name Root"] = root_name
    row.update(canonical_trait_row(chili))
    row.update(dict(zip(EXTRA_COLUMNS, (
        chili.width_bbox, chili.height_bbox, chili.width_fruit_9seg,
        chili.length_fruit_9seg, chili.fruit_body_area, chili.seed_stats["min"],
        chili.seed_stats["max"], chili.seed_stats["avg"], chili.seed_stats["std"],
        chili.redness, chili.color_uniformity, chili.ratio, chili.scale_source,
    ))))
    return row


@lru_cache(maxsize=16)
def _file_hash(path, size, mtime_ns):
    del size, mtime_ns
    digest = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size,
            "sha256": _file_hash(str(path), stat.st_size, stat.st_mtime_ns)}


def create_manifest(config, inputs):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=BASE_DIR, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    versions = {}
    for package in ("torch", "ultralytics", "onnxruntime", "numpy", "opencv-python", "scipy", "scikit-image", "pandas", "Pillow", "gradio", "pyzbar", "shapely", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    # Hash source files too: HEAD alone does not identify an uncommitted run.
    sources = [BASE_DIR / "app.py", BASE_DIR / "predictor.py"] + sorted((BASE_DIR / "src").glob("*.py"))
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "git_revision": revision,
            "git_dirty": dirty, "source_files": [file_identity(p) for p in sources],
            "configuration": config.effective_settings(), "versions": versions,
            "checkpoints": {"segmentation": file_identity(config.weight_seg),
                            "detection": file_identity(config.weight_det) if config.detector_backend == "yolov7_onnx" else None},
            "inputs": [file_identity(p) for p in inputs], "images": [], "errors": []}


def image_diagnostics(fruits):
    return {"input": str(Path(fruits.path).resolve()),
            "stored_image_size": list(fruits.stored_image_size),
            "decoded_image_size": [int(fruits.width_img), int(fruits.height_img)],
            "scale_ratio_mm_per_px": float(fruits.ratio), "scale_source": fruits.scale_source,
            "scale_warning": fruits.scale_warning,
            "candidates": fruits.candidate_diagnostics,
            "fruits": [{"id": c.id, "name": c.name, "diagnostics": c.get_measurement_diagnostics()} for c in fruits.items]}


def write_exports(output_dir, dataframe, manifest, save_csv=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output_dir / "output.csv", output_dir / "manifest.json"
    if save_csv:
        dataframe.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return csv_path, json_path


def save_overlays(fruits, output_dir):
    import cv2
    from src.object import draw_full_mask_with_labels
    output_dir = Path(output_dir) / "images" / Path(fruits.path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    for chili in fruits.items:
        image = draw_full_mask_with_labels(chili.img, chili.polygons)
        path = output_dir / chili.name
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not save overlay: {path}")
