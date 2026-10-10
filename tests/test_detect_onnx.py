import unittest
from types import SimpleNamespace

import numpy as np

from src.detect import (
    DetectModel,
    decode_yolov7_output,
    letterbox,
    restore_boxes,
)


class LetterboxTests(unittest.TestCase):
    def test_horizontal_image_round_trip_and_roi_offset(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, scale, padding = letterbox(image, 640)
        model_box = np.asarray([[64, 192, 576, 448]], dtype=np.float32)

        restored = restore_boxes(
            model_box, scale, padding, image.shape, crop_top=540
        )

        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 3.2)
        self.assertEqual(padding, (0.0, 160.0))
        np.testing.assert_allclose(
            restored[0], [20, 550, 180, 630], atol=1e-5
        )

    def test_vertical_image_round_trip(self):
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        _padded, scale, padding = letterbox(image, 640)
        model_box = np.asarray([[192, 64, 448, 576]], dtype=np.float32)

        restored = restore_boxes(model_box, scale, padding, image.shape)

        self.assertEqual(padding, (160.0, 0.0))
        np.testing.assert_allclose(
            restored[0], [10, 20, 90, 180], atol=1e-5
        )

    def test_restore_clips_boxes_and_empty_input(self):
        boxes = np.asarray([[-20, -10, 700, 680]], dtype=np.float32)
        restored = restore_boxes(
            boxes, 1.0, (0.0, 0.0), (100, 200, 3), crop_top=540
        )
        np.testing.assert_array_equal(restored[0], [0, 540, 200, 640])
        empty = restore_boxes(
            np.empty((0, 4)), 1.0, (0.0, 0.0), (100, 200, 3)
        )
        self.assertEqual(empty.shape, (0, 4))


class DecodeTests(unittest.TestCase):
    def test_filters_seed_low_confidence_and_overlapping_boxes(self):
        output = np.asarray(
            [[
                [100, 100, 40, 40, 0.95, 0.01, 0.99],
                [102, 102, 40, 40, 0.90, 0.01, 0.99],
                [300, 300, 30, 30, 0.99, 0.99, 0.01],
                [400, 400, 30, 30, 0.80, 0.01, 0.99],
                [500, 500, 20, 20, 0.96, 0.01, 0.98],
            ]],
            dtype=np.float32,
        )

        boxes, scores = decode_yolov7_output(
            output, confidence_threshold=0.90, iou_threshold=0.70
        )

        self.assertEqual(boxes.shape, (2, 4))
        self.assertTrue(np.all(scores >= 0.90))
        observed = {tuple(box.tolist()) for box in boxes}
        self.assertEqual(
            observed,
            {(80.0, 80.0, 120.0, 120.0), (490.0, 490.0, 510.0, 510.0)},
        )

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

    def test_no_detection_returns_stable_empty_arrays(self):
        boxes, scores = decode_yolov7_output(
            np.zeros((1, 4, 7), dtype=np.float32)
        )
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))


class _FakeSession:
    def __init__(self, input_shape, output_shape, input_type="tensor(float)"):
        self._inputs = [
            SimpleNamespace(name="images", shape=input_shape, type=input_type)
        ]
        self._outputs = [
            SimpleNamespace(name="output", shape=output_shape, type="tensor(float)")
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


class ModelValidationTests(unittest.TestCase):
    def _model_with_session(self, session):
        model = DetectModel.__new__(DetectModel)
        model.session = session
        model.input_size = 640
        model.input_name = None
        model.output_name = None
        return model

    def test_invalid_input_signature_has_clear_error(self):
        model = self._model_with_session(
            _FakeSession([1, 3, 320, 320], [1, 6300, 7])
        )
        with self.assertRaisesRegex(ValueError, "input must be"):
            model._validate_onnx_signature()

    def test_invalid_output_signature_has_clear_error(self):
        model = self._model_with_session(
            _FakeSession([1, 3, 640, 640], [1, 25200, 6])
        )
        with self.assertRaisesRegex(ValueError, "output must end in 7"):
            model._validate_onnx_signature()

    def test_vertical_sort_uses_y_coordinate_variance(self):
        model = DetectModel(backend="threshold")
        boxes = [[100, 90, 5, 5], [0, 290, 5, 5], [50, 190, 5, 5]]
        self.assertEqual(
            model.sort_bboxes_by_variance(boxes),
            [[100, 90, 5, 5], [50, 190, 5, 5], [0, 290, 5, 5]],
        )


if __name__ == "__main__":
    unittest.main()
