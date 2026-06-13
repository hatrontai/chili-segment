import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import evaluate
from src.debug import CLASS_NAMES, save_chili_debug
from src.measurement import clean_mask, color_metrics, length_metrics


class FakeChili:
    def __init__(self):
        self.img = np.full((100, 160, 3), 235, dtype=np.uint8)
        self.mask = np.zeros((100, 160), dtype=np.uint8)
        cv2.ellipse(self.mask, (80, 50), (55, 24), 0, 0, 360, 255, -1)
        self.img[self.mask > 0] = (220, 35, 25)

        seed = np.zeros_like(self.mask)
        cv2.circle(seed, (70, 50), 7, 255, -1)
        endocarp = np.zeros_like(self.mask)
        cv2.ellipse(endocarp, (80, 50), (45, 18), 0, 0, 360, 255, -1)
        self.rseg = {
            "masks_data": [
                {"class_id": 0, "mask": seed},
                {"class_id": 3, "mask": endocarp},
            ]
        }

        self.id = 0
        self.name = "sample_0.jpg"
        self.bbox = [20, 30, 160, 100]
        self.width_bbox = 40.0
        self.height_bbox = 80.0
        self.width_fruit = 30.0
        self.length_fruit = 75.0
        self.area_fruit = 2500.0
        self.redness = 0.8
        self.wrinkle = 0.1
        self.number_seeds = 1
        self.ratio = 0.2


class DebugOutputTest(unittest.TestCase):
    def test_saves_all_debug_artifacts(self):
        chili = FakeChili()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sample" / "chili_000"
            row = save_chili_debug(chili, "sample.jpg", output_dir)

            expected = [
                "step1_debug/crop.jpg",
                "step1_debug/raw_mask_overlay.jpg",
                "step2_masks/fruit_mask_before.png",
                "step2_masks/fruit_mask_after.png",
                "step3_length/skeleton_pruned.png",
                "step3_length/skeleton_path_overlay.jpg",
                "step4_color/color_overlay.jpg",
                "step5_measurement/calibrated_measurements.txt",
                "debug_summary.jpg",
            ]
            for relative_path in expected:
                self.assertTrue((output_dir / relative_path).is_file())

            for class_id, class_name in CLASS_NAMES.items():
                before = (
                    output_dir / "step2_masks"
                    / f"{class_id:02d}_{class_name}_before.png"
                )
                after = (
                    output_dir / "step2_masks"
                    / f"{class_id:02d}_{class_name}_after.png"
                )
                self.assertTrue(before.is_file())
                self.assertTrue(after.is_file())

            self.assertGreater(row["seed_mask_area_clean_px"], 0)
            self.assertGreater(row["endocarp_mask_area_clean_px"], 0)
            self.assertEqual(row["outer_mask_area_clean_px"], 0)
            self.assertGreater(row["skeleton_length_px"], 0)
            self.assertGreater(row["red_ratio_robust"], 0.9)

    def test_mask_cleanup_length_color_and_calibration(self):
        mask = np.zeros((100, 160), dtype=np.uint8)
        cv2.ellipse(mask, (80, 50), (55, 20), 0, 0, 360, 255, -1)
        cv2.circle(mask, (80, 50), 6, 0, -1)
        cv2.circle(mask, (5, 5), 2, 255, -1)

        cleaned = clean_mask(mask, min_area=100, keep_largest=True)
        self.assertEqual(cleaned[50, 80], 255)
        self.assertEqual(cleaned[5, 5], 0)

        lengths = length_metrics(cleaned, min_branch_length=8)
        self.assertGreater(lengths["skeleton_length_px"], 80)
        self.assertGreater(lengths["contour_perimeter_px"], 0)

        image = np.full((100, 160, 3), 240, dtype=np.uint8)
        image[cleaned > 0] = (230, 30, 20)
        colors = color_metrics(image, cleaned)
        self.assertGreater(colors["red_ratio_robust"], 0.95)

        with tempfile.TemporaryDirectory() as temp_dir:
            row = save_chili_debug(
                FakeChili(),
                "sample.jpg",
                Path(temp_dir),
                pixel_per_cm=10,
                calibration_source="pixel_per_cm",
            )
            self.assertAlmostEqual(
                row["skeleton_length_cm"],
                row["skeleton_length_px"] / 10,
                places=4,
            )
            self.assertNotEqual(row["fruit_area_cm2"], "")

    def test_evaluation_writes_feature_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            image_path = input_dir / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))

            args = Namespace(
                input_path=str(input_dir),
                output_dir=str(output_dir),
                weight_seg="unused.pt",
                device="cpu",
                use_model_det=False,
                weight_det=None,
                recursive=False,
                fail_fast=True,
                min_component_area=100,
                min_branch_length=12.0,
                pixel_per_cm=10.0,
                reference_length_px=None,
                reference_length_cm=None,
            )
            fake_fruits = type("FakeFruits", (), {"items": [FakeChili()]})()

            with patch.object(evaluate, "build_models", return_value=(object(), object())):
                with patch.object(evaluate, "create_fruits", return_value=fake_fruits):
                    results = evaluate.run_evaluation(args)

            self.assertEqual(len(results), 1)
            self.assertTrue((output_dir / "features.csv").is_file())
            self.assertTrue(
                (output_dir / "sample" / "chili_000" / "debug_summary.jpg").is_file()
            )
            self.assertTrue(
                (output_dir / "sample" / "step1_detection_bboxes.jpg").is_file()
            )
            self.assertEqual(results.iloc[0]["pixel_per_cm"], 10.0)

    def test_reference_calibration(self):
        args = Namespace(
            pixel_per_cm=None,
            reference_length_px=250.0,
            reference_length_cm=5.0,
        )
        scale, source = evaluate.resolve_calibration(args)
        self.assertEqual(scale, 50.0)
        self.assertEqual(source, "reference_object")


if __name__ == "__main__":
    unittest.main()
