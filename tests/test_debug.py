import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import evaluate
from src.debug import CLASS_NAMES, save_chili_debug


class FakeChili:
    def __init__(self):
        self.img = np.full((100, 160, 3), 235, dtype=np.uint8)
        self.mask = np.zeros((100, 160), dtype=np.uint8)
        cv2.ellipse(self.mask, (80, 50), (55, 24), 0, 0, 360, 255, -1)

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
                "crop.jpg",
                "feature_mask.png",
                "skeleton.png",
                "mask_overlay.jpg",
                "skeleton_overlay.jpg",
                "debug_summary.jpg",
            ]
            for relative_path in expected:
                self.assertTrue((output_dir / relative_path).is_file())

            for class_id, class_name in CLASS_NAMES.items():
                mask_path = output_dir / "masks" / f"{class_id:02d}_{class_name}.png"
                self.assertTrue(mask_path.is_file())

            self.assertGreater(row["seed_mask_area_px"], 0)
            self.assertGreater(row["endocarp_mask_area_px"], 0)
            self.assertEqual(row["outer_mask_area_px"], 0)

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


if __name__ == "__main__":
    unittest.main()
