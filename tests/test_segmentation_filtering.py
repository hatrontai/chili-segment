import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.object import Chili, Fruits, InvalidChiliCandidate
from src.segment import SegmentModel


class _CpuArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeMasks:
    def __init__(self, masks):
        self.data = [_CpuArray(mask) for mask in masks]

    def __len__(self):
        return len(self.data)


class _FakeSegmentationModel:
    def __init__(self, class_ids, masks):
        boxes = SimpleNamespace(
            cls=np.asarray(class_ids),
            conf=np.full(len(class_ids), 0.9),
            xyxy=[
                _CpuArray([0, 0, mask.shape[1], mask.shape[0]])
                for mask in masks
            ],
        )
        result = SimpleNamespace(boxes=boxes, masks=_FakeMasks(masks))
        self.result = result

    def predict(self, **_kwargs):
        return [self.result]


def _segmenter(class_ids, masks, chili_method="threshold"):
    segmenter = SegmentModel.__new__(SegmentModel)
    segmenter.model = _FakeSegmentationModel(class_ids, masks)
    segmenter.device = "cpu"
    segmenter.imgsz = 640
    segmenter.chili_method = chili_method
    segmenter.chili_threshold = 16
    return segmenter


class SegmentationCandidateTests(unittest.TestCase):
    @staticmethod
    def _threshold_image():
        image = np.full((80, 120, 3), 10, dtype=np.uint8)
        image[10:65, 20:70, 0] = 240
        image[25:40, 102:118, 0] = 240
        return image

    def test_threshold_region_without_yolo_chili_is_rejected(self):
        image = self._threshold_image()
        outer_mask = np.ones(image.shape[:2], dtype=np.float32)
        result = _segmenter([1], [outer_mask]).predict(image)

        self.assertFalse(result["stats"]["is_valid_chili"])
        self.assertEqual(
            result["stats"]["rejection_reason"], "no_yolo_chili_mask"
        )
        self.assertEqual(result["masks_data"], [])
        self.assertEqual(result["polygons"], [])

    def test_threshold_keeps_largest_contour_and_clips_subordinate_masks(self):
        image = self._threshold_image()
        yolo_chili = np.zeros(image.shape[:2], dtype=np.float32)
        yolo_chili[10:65, 20:70] = 1
        outer_mask = np.ones(image.shape[:2], dtype=np.float32)

        result = _segmenter(
            [4, 1], [yolo_chili, outer_mask]
        ).predict(image)
        masks = {
            item["class_id"]: item["mask"] for item in result["masks_data"]
        }
        chili_mask = masks[4]

        self.assertTrue(result["stats"]["is_valid_chili"])
        self.assertGreater(np.count_nonzero(chili_mask[:, 20:70]), 0)
        self.assertEqual(np.count_nonzero(chili_mask[:, 95:]), 0)
        self.assertFalse(bool(np.any((masks[1] > 0) & (chili_mask == 0))))

    def test_yolo_mode_keeps_largest_chili_instance(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        small = np.zeros(image.shape[:2], dtype=np.float32)
        small[10:20, 5:15] = 1
        large = np.zeros(image.shape[:2], dtype=np.float32)
        large[20:65, 40:95] = 1

        result = _segmenter(
            [4, 4], [small, large], chili_method="yolo"
        ).predict(image)
        chili_masks = [
            item["mask"]
            for item in result["masks_data"]
            if item["class_id"] == 4
        ]

        self.assertTrue(result["stats"]["is_valid_chili"])
        self.assertEqual(len(chili_masks), 1)
        self.assertEqual(np.count_nonzero(chili_masks[0][:, :25]), 0)
        self.assertGreater(np.count_nonzero(chili_masks[0][:, 40:95]), 0)


class ChiliCropTests(unittest.TestCase):
    def test_pixels_outside_selected_chili_are_black(self):
        chili = Chili.__new__(Chili)
        chili.img = np.full((8, 10, 3), 200, dtype=np.uint8)
        chili.mask_chili = np.zeros((8, 10), dtype=np.uint8)
        chili.mask_chili[2:7, 3:8] = 255

        chili._black_outside_chili_mask()

        self.assertTrue(np.all(chili.img[chili.mask_chili == 0] == 0))
        self.assertTrue(np.all(chili.img[chili.mask_chili > 0] == 200))


class _DetectionModel:
    def predict(self, _path):
        return [
            [30, 0, 5, 5],
            [10, 0, 5, 5],
            [20, 0, 5, 5],
        ]


class FruitCollectionTests(unittest.TestCase):
    def test_invalid_candidates_are_removed_and_valid_ids_are_contiguous(self):
        def create_chili(_path, index, _model_segment, bbox, _ratio, **_kwargs):
            if bbox[0] == 20:
                raise InvalidChiliCandidate("no_yolo_chili_mask")
            return SimpleNamespace(id=index, bbox=bbox)

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "plain.png"
            Image.new("RGB", (50, 50), "black").save(image_path)
            with patch.object(Fruits, "qr_code", return_value=None), patch(
                "src.object.Chili", side_effect=create_chili
            ):
                fruits = Fruits(
                    str(image_path),
                    model_detect=_DetectionModel(),
                    model_segment=object(),
                )
                fruits.image.close()

        self.assertEqual([bbox[0] for bbox in fruits.box_cut], [10, 30])
        self.assertEqual([fruit.id for fruit in fruits.items], [0, 1])
        self.assertEqual(fruits.num_fruit, 2)


if __name__ == "__main__":
    unittest.main()
