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


def _segmenter(class_ids, masks, confidences=None):
    segmenter = SegmentModel.__new__(SegmentModel)
    segmenter.model = _FakeSegmentationModel(class_ids, masks, confidences)
    segmenter.device = "cpu"
    segmenter.imgsz = 640
    segmenter.MODEL_CONF_THRESHOLD = 0.01
    segmenter.fruit_threshold_offset = 16
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
        self.assertTrue(result["stats"]["is_valid_fruit"])
        self.assertFalse(result["stats"]["has_any_segment_mask"])
        self.assertEqual(result["stats"]["validation_mode"], "detector_trusted")
        self.assertEqual(result["stats"]["fruit_mask_source"], "red_channel_threshold")
        self.assertGreater(np.count_nonzero(result["fruit_mask"]), 0)
        self.assertEqual(result["masks_data"], [])

    def test_threshold_detector_accepts_any_segment_class(self):
        image = self._threshold_image()
        outer_mask = np.ones(image.shape[:2], dtype=np.float32)
        result = _segmenter([1], [outer_mask]).predict(
            image, validation_mode="any_segment"
        )

        self.assertTrue(result["stats"]["is_valid_fruit"])
        self.assertTrue(result["stats"]["has_any_segment_mask"])
        self.assertIsNone(result["stats"]["rejection_reason"])

    def test_threshold_detector_rejects_when_no_segment_mask_exists(self):
        image = self._threshold_image()
        result = _segmenter([], []).predict(
            image, validation_mode="any_segment"
        )

        self.assertFalse(result["stats"]["is_valid_fruit"])
        self.assertEqual(result["stats"]["rejection_reason"], "no_segment_mask")
        self.assertEqual(result["masks_data"], [])
        self.assertEqual(result["polygons"], [])

    def test_threshold_keeps_largest_contour_and_clips_subordinate_masks(self):
        image = self._threshold_image()
        outer_mask = np.ones(image.shape[:2], dtype=np.float32)

        result = _segmenter([1], [outer_mask]).predict(image)
        masks = {
            item["class_id"]: item["mask"] for item in result["masks_data"]
        }
        fruit_mask = result["fruit_mask"]

        self.assertTrue(result["stats"]["is_valid_fruit"])
        self.assertGreater(np.count_nonzero(fruit_mask[:, 20:70]), 0)
        self.assertEqual(np.count_nonzero(fruit_mask[:, 95:]), 0)
        self.assertFalse(bool(np.any((masks[1] > 0) & (fruit_mask == 0))))

    def test_legacy_class_four_is_ignored_and_low_component_is_filtered(self):
        image = self._threshold_image()
        legacy_class_four = np.ones(image.shape[:2], dtype=np.float32)
        outer = np.ones(image.shape[:2], dtype=np.float32)

        result = _segmenter(
            [4, 1], [legacy_class_four, outer], confidences=[0.9, 0.02]
        ).predict(image)
        class_ids = [item["class_id"] for item in result["masks_data"]]

        self.assertTrue(result["stats"]["is_valid_fruit"])
        self.assertNotIn(4, class_ids)
        self.assertNotIn(1, class_ids)

    def test_below_threshold_segment_does_not_confirm_threshold_detector(self):
        image = self._threshold_image()
        legacy_class_four = np.zeros(image.shape[:2], dtype=np.float32)
        legacy_class_four[10:65, 20:70] = 1

        result = _segmenter([4], [legacy_class_four], confidences=[0.9]).predict(
            image, validation_mode="any_segment"
        )

        self.assertFalse(result["stats"]["is_valid_fruit"])
        self.assertEqual(result["stats"]["rejection_reason"], "no_segment_mask")

    def test_class_four_never_enters_segmentation_outputs(self):
        image = self._threshold_image()
        class_four_masks = [
            np.ones(image.shape[:2], dtype=np.float32),
            np.ones(image.shape[:2], dtype=np.float32),
        ]
        result = _segmenter([4, 4], class_four_masks).predict(image)

        self.assertTrue(result["stats"]["is_valid_fruit"])
        self.assertEqual(result["masks_data"], [])
        self.assertEqual(result["polygons"], [])
        self.assertGreater(np.count_nonzero(result["fruit_mask"]), 0)


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
    def test_pixels_outside_threshold_fruit_mask_are_black(self):
        chili = Chili.__new__(Chili)
        chili.img = np.full((8, 10, 3), 200, dtype=np.uint8)
        chili.mask_fruit = np.zeros((8, 10), dtype=np.uint8)
        chili.mask_fruit[2:7, 3:8] = 255

        chili._black_outside_fruit_mask()

        self.assertTrue(np.all(chili.img[chili.mask_fruit == 0] == 0))
        self.assertTrue(np.all(chili.img[chili.mask_fruit > 0] == 200))


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
