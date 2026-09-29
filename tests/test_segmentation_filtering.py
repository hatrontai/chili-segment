import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.detect import decode_yolov7_output
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
    def __init__(self, class_ids, masks, confidences=None):
        if confidences is None:
            confidences = np.full(len(class_ids), 0.9)
        boxes = SimpleNamespace(
            cls=np.asarray(class_ids),
            conf=np.asarray(confidences),
            xyxy=[
                _CpuArray([0, 0, mask.shape[1], mask.shape[0]])
                for mask in masks
            ],
        )
        result = SimpleNamespace(boxes=boxes, masks=_FakeMasks(masks))
        self.result = result

    def predict(self, **_kwargs):
        return [self.result]


def _segmenter(class_ids, masks, chili_method="threshold", confidences=None):
    segmenter = SegmentModel.__new__(SegmentModel)
    segmenter.model = _FakeSegmentationModel(class_ids, masks, confidences)
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

    def test_model_detector_trusts_candidate_without_segment_masks(self):
        image = self._threshold_image()
        result = _segmenter([], []).predict(
            image, validation_mode="detector_trusted"
        )
        masks = {
            item["class_id"]: item["mask"] for item in result["masks_data"]
        }

        self.assertTrue(result["stats"]["is_valid_chili"])
        self.assertFalse(result["stats"]["has_any_segment_mask"])
        self.assertEqual(result["stats"]["validation_mode"], "detector_trusted")
        self.assertEqual(result["stats"]["mask_source"], "threshold")
        self.assertGreater(np.count_nonzero(masks[4]), 0)

    def test_threshold_detector_accepts_any_segment_class(self):
        image = self._threshold_image()
        outer_mask = np.ones(image.shape[:2], dtype=np.float32)
        result = _segmenter([1], [outer_mask]).predict(
            image, validation_mode="any_segment"
        )

        self.assertTrue(result["stats"]["is_valid_chili"])
        self.assertTrue(result["stats"]["has_any_segment_mask"])
        self.assertIsNone(result["stats"]["rejection_reason"])

    def test_threshold_detector_rejects_when_no_segment_mask_exists(self):
        image = self._threshold_image()
        result = _segmenter([], []).predict(
            image, validation_mode="any_segment"
        )

        self.assertFalse(result["stats"]["is_valid_chili"])
        self.assertEqual(result["stats"]["rejection_reason"], "no_segment_mask")
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

    def test_low_confidence_chili_validates_without_admitting_low_component(self):
        image = self._threshold_image()
        chili = np.zeros(image.shape[:2], dtype=np.float32)
        chili[10:65, 20:70] = 1
        outer = np.ones(image.shape[:2], dtype=np.float32)

        result = _segmenter(
            [4, 1], [chili, outer], confidences=[0.02, 0.02]
        ).predict(image)
        class_ids = [item["class_id"] for item in result["masks_data"]]

        self.assertTrue(result["stats"]["is_valid_chili"])
        self.assertIn(4, class_ids)
        self.assertNotIn(1, class_ids)

    def test_below_threshold_segment_does_not_confirm_threshold_detector(self):
        image = self._threshold_image()
        chili = np.zeros(image.shape[:2], dtype=np.float32)
        chili[10:65, 20:70] = 1

        result = _segmenter([4], [chili], confidences=[0.005]).predict(
            image, validation_mode="any_segment"
        )

        self.assertFalse(result["stats"]["is_valid_chili"])
        self.assertEqual(result["stats"]["rejection_reason"], "no_segment_mask")

    def test_yolo_mode_falls_back_to_threshold_without_chili_class(self):
        image = self._threshold_image()
        endocarp = np.zeros(image.shape[:2], dtype=np.float32)
        endocarp[20:50, 30:60] = 1

        result = _segmenter(
            [3], [endocarp], chili_method="yolo"
        ).predict(image, validation_mode="detector_trusted")
        masks = {
            item["class_id"]: item["mask"] for item in result["masks_data"]
        }

        self.assertTrue(result["stats"]["is_valid_chili"])
        self.assertEqual(result["stats"]["mask_source"], "threshold_fallback")
        self.assertGreater(np.count_nonzero(masks[4]), 0)
        self.assertFalse(bool(np.any((masks[3] > 0) & (masks[4] == 0))))

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


class DetectorDeduplicationTests(unittest.TestCase):
    def test_nested_duplicate_is_suppressed_by_containment(self):
        output = np.asarray(
            [[
                [100, 100, 100, 100, 0.95, 0.01, 0.99],
                [100, 100, 50, 50, 0.90, 0.01, 0.99],
            ]],
            dtype=np.float32,
        )

        boxes, _scores = decode_yolov7_output(
            output,
            confidence_threshold=0.75,
            iou_threshold=0.50,
            containment_threshold=0.80,
        )

        self.assertEqual(boxes.shape, (1, 4))
        np.testing.assert_array_equal(boxes[0], [50, 50, 150, 150])

    def test_stricter_iou_keeps_only_higher_confidence_box(self):
        output = np.asarray(
            [[
                [100, 100, 100, 100, 0.95, 0.01, 0.99],
                [140, 100, 100, 100, 0.85, 0.01, 0.99],
            ]],
            dtype=np.float32,
        )

        boxes, scores = decode_yolov7_output(
            output,
            confidence_threshold=0.70,
            iou_threshold=0.30,
            containment_threshold=0.60,
        )

        self.assertEqual(boxes.shape, (1, 4))
        self.assertGreater(float(scores[0]), 0.90)

    def test_nearby_non_duplicate_boxes_are_both_kept(self):
        output = np.asarray(
            [[
                [100, 100, 100, 100, 0.95, 0.01, 0.99],
                [170, 100, 100, 100, 0.90, 0.01, 0.99],
            ]],
            dtype=np.float32,
        )

        boxes, _scores = decode_yolov7_output(
            output,
            confidence_threshold=0.75,
            iou_threshold=0.50,
            containment_threshold=0.80,
        )

        self.assertEqual(boxes.shape, (2, 4))


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
    backend = "threshold"

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
