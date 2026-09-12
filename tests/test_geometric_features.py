import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from src.geometric_features import (
    extract_outer_edge_runs,
    longest_backbone_skeleton,
    measure_skeleton_point,
    sample_skeleton_at_spacing,
    segment_polyline_by_exact_chord,
)
from src.object import DEFAULT_DATASET_RATIO_MM_PER_PX, Fruits


class _NoDetectionModel:
    def predict(self, _path):
        return []


class ScaleFallbackTests(unittest.TestCase):
    def test_missing_qr_uses_documented_dataset_mean(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "plain.png"
            Image.new("RGB", (20, 20), "black").save(image_path)
            with patch.object(Fruits, "qr_code", return_value=None):
                fruits = Fruits(
                    str(image_path),
                    model_detect=_NoDetectionModel(),
                    model_segment=None,
                )
                fruits.image.close()

        self.assertAlmostEqual(fruits.ratio, DEFAULT_DATASET_RATIO_MM_PER_PX)
        self.assertEqual(fruits.scale_source, "dataset_mean_fallback")
        self.assertIn("QR code not found", fruits.scale_warning)


class SkeletonSamplingTests(unittest.TestCase):
    def test_open_path_uses_one_mm_spacing_and_keeps_tail_endpoint(self):
        skeleton = np.zeros((20, 50), dtype=bool)
        skeleton[10, 10:36] = True
        samples = sample_skeleton_at_spacing(skeleton, ratio=0.1, spacing_mm=1.0)

        np.testing.assert_allclose(
            samples.previous_gap_mm.dropna().to_numpy(),
            [1.0, 1.0, 0.5],
            atol=0.01,
        )
        self.assertEqual(int(samples.is_endpoint.sum()), 2)
        self.assertTrue(bool(samples.iloc[-1].is_tail_gap))

    def test_short_path_has_only_two_endpoints(self):
        skeleton = np.zeros((10, 10), dtype=bool)
        skeleton[5, 2:7] = True
        samples = sample_skeleton_at_spacing(skeleton, ratio=0.1, spacing_mm=1.0)

        self.assertEqual(len(samples), 2)
        self.assertTrue(bool(samples.is_endpoint.all()))
        self.assertAlmostEqual(float(samples.iloc[-1].previous_gap_mm), 0.4)

    def test_diagonal_and_multiple_components_are_sampled_independently(self):
        skeleton = np.zeros((45, 45), dtype=bool)
        np.fill_diagonal(skeleton[5:26, 5:26], True)
        skeleton[35, 5:31] = True
        samples = sample_skeleton_at_spacing(skeleton, ratio=0.05, spacing_mm=1.0)

        self.assertEqual(int(samples.component_id.nunique()), 2)
        regular = samples.loc[
            samples.previous_gap_mm.notna() & ~samples.is_tail_gap,
            "previous_gap_mm",
        ]
        np.testing.assert_allclose(regular, 1.0, atol=0.01)

    def test_closed_path_does_not_duplicate_start(self):
        skeleton = np.zeros((30, 30), dtype=bool)
        skeleton[5, 5:25] = skeleton[24, 5:25] = True
        skeleton[5:25, 5] = skeleton[5:25, 24] = True
        samples = sample_skeleton_at_spacing(skeleton, ratio=0.1, spacing_mm=1.0)

        self.assertTrue(bool(samples.is_closed_component.all()))
        self.assertFalse(
            bool(samples[["skeleton_x", "skeleton_y"]].duplicated().any())
        )

    def test_t_branch_keeps_longest_backbone(self):
        skeleton = np.zeros((101, 101), dtype=bool)
        skeleton[50, 10:91] = True
        skeleton[50:71, 50] = True
        backbone = longest_backbone_skeleton(skeleton)

        self.assertGreaterEqual(int(np.count_nonzero(backbone[50, 10:91])), 79)
        self.assertEqual(int(np.count_nonzero(backbone[52:71, 50])), 0)


class ThicknessTests(unittest.TestCase):
    @staticmethod
    def synthetic_wall(angle_deg=0.0):
        size = 121
        yy, xx = np.mgrid[:size, :size]
        angle = np.deg2rad(angle_deg)
        normal = np.array([-np.sin(angle), np.cos(angle)])
        signed_normal = (xx - 60) * normal[0] + (yy - 60) * normal[1]
        wall = np.abs(signed_normal) <= 5.0
        inner_reference = signed_normal < -5.0
        outer_reference = signed_normal > 5.0
        return {
            "wall": wall,
            "distance_to_inner": ndi.distance_transform_edt(~inner_reference),
            "distance_to_outer": ndi.distance_transform_edt(~outer_reference),
            "signed_distance": ndi.distance_transform_edt(wall)
            - ndi.distance_transform_edt(~wall),
        }

    def test_straight_and_diagonal_wall(self):
        for angle, expected_px in ((0.0, 11.0), (27.0, 10.0)):
            with self.subTest(angle=angle):
                result = measure_skeleton_point(
                    (60.0, 60.0), self.synthetic_wall(angle)
                )
                self.assertIsNotNone(result)
                self.assertLessEqual(abs(result["length_px"] - expected_px), 0.3)

    def test_curved_annulus_crosses_both_edges(self):
        yy, xx = np.mgrid[:151, :151]
        radius = np.hypot(xx - 75, yy - 75)
        wall = (radius >= 35) & (radius <= 45)
        data = {
            "wall": wall,
            "distance_to_inner": ndi.distance_transform_edt(~(radius < 35)),
            "distance_to_outer": ndi.distance_transform_edt(~(radius > 45)),
            "signed_distance": ndi.distance_transform_edt(wall)
            - ndi.distance_transform_edt(~wall),
        }
        result = measure_skeleton_point((115.0, 75.0), data)

        self.assertIsNotNone(result)
        # The analytical annulus is 10 px thick; raster signed-distance
        # interpolation places the two zero crossings slightly outside it.
        self.assertLessEqual(abs(result["length_px"] - 10.25), 0.35)
        inner_radius = np.linalg.norm(result["inner"] - np.array([75.0, 75.0]))
        outer_radius = np.linalg.norm(result["outer"] - np.array([75.0, 75.0]))
        self.assertLess(inner_radius, 35.0)
        self.assertGreater(outer_radius, 44.0)


class WrinkleTests(unittest.TestCase):
    def test_straight_is_zero_and_zigzag_is_larger(self):
        straight = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
        zigzag = np.array(
            [
                [0, 0],
                [10, 7],
                [20, 0],
                [30, 7],
                [40, 0],
                [50, 7],
                [60, 0],
                [70, 7],
                [80, 0],
            ],
            dtype=float,
        )
        straight_segments = segment_polyline_by_exact_chord(straight, 20.0)[2]
        zigzag_segments = segment_polyline_by_exact_chord(zigzag, 20.0)[2]
        straight_wrinkle = np.mean(
            [segment["curvature_excess"] for segment in straight_segments]
        )
        zigzag_wrinkle = np.mean(
            [segment["curvature_excess"] for segment in zigzag_segments]
        )

        self.assertAlmostEqual(float(straight_wrinkle), 0.0)
        self.assertGreater(float(zigzag_wrinkle), float(straight_wrinkle))

    def test_interpolated_chords_do_not_exceed_threshold(self):
        contour = np.array(
            [[0, 0], [15, 5], [30, -3], [45, 6], [60, 0]], dtype=float
        )
        segments = segment_polyline_by_exact_chord(contour, 20.0)[2]

        self.assertTrue(segments)
        self.assertLessEqual(
            max(segment["chord_px"] for segment in segments), 20.0 + 0.01
        )

    def test_outer_runs_never_include_stem_exclusion(self):
        wall = np.zeros((80, 100), dtype=bool)
        wall[10:70, 10:90] = True
        inner_reference = np.zeros_like(wall)
        inner_reference[20:60, 20:80] = True
        outer_reference = ~wall
        stem_exclusion = np.zeros_like(wall)
        stem_exclusion[:18, 40:61] = True
        data = {
            "wall": wall,
            "distance_to_inner": ndi.distance_transform_edt(~inner_reference),
            "distance_to_outer": ndi.distance_transform_edt(~outer_reference),
            "stem_exclusion": stem_exclusion,
        }

        for run in extract_outer_edge_runs(data):
            xy = np.rint(run).astype(int)
            self.assertFalse(bool(stem_exclusion[xy[:, 1], xy[:, 0]].any()))


if __name__ == "__main__":
    unittest.main()
