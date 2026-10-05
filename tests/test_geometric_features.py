import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from src.geometric_features import (
    build_mesocarp_data,
    calculate_mesocarp_thickness_stats,
    extract_chili_body_contours,
    longest_backbone_skeleton,
    measure_skeleton_point,
    sample_skeleton_at_spacing,
    segment_polyline_by_exact_chord,
)
from src.object import CLASS_NAMES, Chili, DEFAULT_DATASET_RATIO_MM_PER_PX, Fruits
from src.traits import CANONICAL_TRAITS, CANONICAL_TRAIT_COLUMNS, canonical_trait_row


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


class TerminologyTests(unittest.TestCase):
    def test_class_two_is_pcp_without_changing_its_id(self):
        self.assertEqual(CLASS_NAMES[2], "PCP")

    def test_canonical_trait_registry_has_expected_symbols_and_units(self):
        expected = [
            "FL", "FW_mean", "FW_min", "FW_max", "FA", "MA",
            "MT_mean", "MT_min", "MT_max", "PA", "SA", "SC",
            "MAF", "SCR", "FCI", "FAI", "MWI",
        ]
        self.assertEqual([item.symbol for item in CANONICAL_TRAITS], expected)
        self.assertEqual(len(set(expected)), 17)
        units = {item.symbol: item.unit for item in CANONICAL_TRAITS}
        self.assertEqual(units["FL"], "mm")
        self.assertEqual(units["FA"], "mm^2")
        self.assertEqual(units["SC"], "seed")
        self.assertEqual(units["FCI"], "dimensionless")
        self.assertEqual(len(CANONICAL_TRAIT_COLUMNS), 17)

    def test_chili_exposes_all_canonical_values_without_removing_legacy_names(self):
        chili = Chili.__new__(Chili)
        chili.length_fruit = 10.0
        chili.width_fruit = 4.0
        chili.width_fruit_stats = {"min": 2.0, "max": 6.0, "avg": 4.0}
        chili.area_fruit = 30.0
        chili.mesocarp_area = 12.0
        chili.mesocarp_thickness_stats = {
            "min_mesocarp_thickness": 1.0,
            "avg_mesocarp_thickness": 2.0,
            "max_mesocarp_thickness": 3.0,
        }
        chili.placenta_area = 5.0
        chili.seed_stats = {
            "sum": 7.0, "min": 1.0, "max": 3.0, "avg": 2.0, "std": 0.5,
        }
        chili.number_seeds = 4
        chili.flesh_placenta_ratio = 2.4
        chili.mesocarp_area_fraction = 0.6
        chili.seed_density = 0.25
        chili.curvature_index = 0.9
        chili.asymmetry_index = 0.1
        chili.mesocarp_wrinkle_index = 0.05

        values = chili.get_trait_values()
        row = canonical_trait_row(chili)

        self.assertEqual(list(values), [item.symbol for item in CANONICAL_TRAITS])
        self.assertEqual(values["FL"], chili.length_fruit)
        self.assertEqual(values["SA"], chili.seed_stats["sum"])
        self.assertEqual(values["MAF"], 0.6)
        self.assertNotIn("FPR", values)
        self.assertEqual(values["SCR"], chili.seed_density)
        self.assertEqual(values["FCI"], chili.curvature_index)
        self.assertEqual(list(row), list(CANONICAL_TRAIT_COLUMNS))

    def test_maf_excludes_outer_and_pedicel_and_is_scale_independent(self):
        chili = Chili.__new__(Chili)
        chili.mask_chili = np.full((10, 10), 255, dtype=np.uint8)
        chili.mask_outer = np.zeros((10, 10), dtype=np.uint8)
        chili.mask_outer[0, :] = 255
        chili.mask_pedicel = np.zeros((10, 10), dtype=np.uint8)
        chili.mask_pedicel[1, :] = 255
        chili.mask_mesocarp = np.zeros((10, 10), dtype=np.uint8)
        chili.mask_mesocarp[2:6, :] = 255
        for ratio in (0.1, 0.0001, 2.0):
            chili.ratio = ratio
            self.assertEqual(chili.get_mesocarp_area_fraction(), 0.5)
        chili.mask_mesocarp[:] = 0
        self.assertEqual(chili.get_mesocarp_area_fraction(), 0.0)
        chili.mask_mesocarp[2:, :] = 255
        self.assertEqual(chili.get_mesocarp_area_fraction(), 1.0)
        chili.mask_chili[:] = 0
        chili.mask_mesocarp[:] = 0
        self.assertIsNone(chili.get_mesocarp_area_fraction())

    def test_scr_uses_seed_area_over_cavity_union(self):
        chili = Chili.__new__(Chili)
        chili.mask_seed = np.zeros((4, 4), dtype=np.uint8)
        chili.mask_endocarp = np.zeros((4, 4), dtype=np.uint8)
        chili.mask_placenta = np.zeros((4, 4), dtype=np.uint8)
        chili.mask_seed[0, :2] = 255
        chili.mask_endocarp[:2, :] = 255
        chili.mask_placenta[2, :2] = 255
        self.assertEqual(chili.get_seed_density(), 0.2)

    def test_fci_is_chord_over_arc_and_one_means_straight(self):
        chili = Chili.__new__(Chili)
        chili.ratio = 0.1
        chili.spine = {
            "coeffs": np.array([0.0, 0.0, 0.0, 0.0]),
            "y_min": 0,
            "y_max": 10,
        }
        chili.length_fruit = 1.0
        self.assertEqual(chili.get_curvature_index(), 1.0)
        chili.length_fruit = 2.0
        self.assertEqual(chili.get_curvature_index(), 0.5)

    def test_mesocarp_excludes_all_non_mesocarp_classes(self):
        shape = (60, 80)
        chili = type("SyntheticChili", (), {})()
        chili.mask_chili = np.full(shape, 255, dtype=np.uint8)
        chili.mask_endocarp = np.zeros(shape, dtype=np.uint8)
        chili.mask_outer = np.zeros(shape, dtype=np.uint8)
        chili.mask_pcp = np.zeros(shape, dtype=np.uint8)
        chili.mask_seed = np.zeros(shape, dtype=np.uint8)
        chili.mask_pedicel = np.zeros(shape, dtype=np.uint8)
        chili.mask_endocarp[20:40, 20:35] = 255
        chili.mask_outer[:, :5] = 255
        chili.mask_pcp[15:45, 40:55] = 255
        chili.mask_seed[30:35, 60:65] = 255
        excluded = (
            (chili.mask_endocarp > 0)
            | (chili.mask_outer > 0)
            | (chili.mask_pcp > 0)
            | (chili.mask_seed > 0)
        )
        chili.mask_mesocarp = (~excluded).astype(np.uint8) * 255

        data = build_mesocarp_data(chili, ratio=0.1)

        self.assertFalse(bool(np.any(data["mesocarp"] & excluded)))

    def test_pcp_is_partitioned_into_pedicel_region_and_placenta(self):
        chili = Chili.__new__(Chili)
        chili.mask_pcp = np.zeros((10, 10), dtype=np.uint8)
        chili.mask_pcp[1:9, 2:8] = 255
        pedicel = np.zeros_like(chili.mask_pcp)
        pedicel[1:4, 2:8] = 255
        placenta = chili.mask_pcp.copy()
        placenta[1:4, 2:8] = 0
        case1_result = (
            pedicel,
            placenta,
            None,
            None,
            None,
            [],
            True,
            "CASE1_OK",
        )

        with patch("src.object.run_case1_armpit", return_value=case1_result):
            mask_pedicel, mask_placenta = chili._separate_pedicel_and_placenta()

        np.testing.assert_array_equal(
            chili.mask_pcp, np.bitwise_or(mask_pedicel, mask_placenta)
        )
        self.assertFalse(bool(np.any((mask_pedicel > 0) & (mask_placenta > 0))))

    def test_flesh_placenta_ratio_requires_ten_clean_placenta_pixels(self):
        chili = Chili.__new__(Chili)
        chili.ratio = 0.1
        chili.mask_chili = np.full((10, 10), 255, dtype=np.uint8)
        chili.mask_outer = np.zeros((10, 10), dtype=np.uint8)
        chili.mask_outer[0, :] = 255
        chili.mask_pcp = np.zeros((10, 10), dtype=np.uint8)
        chili.mask_pcp[1, :] = 255
        chili.mask_seed = np.zeros((10, 10), dtype=np.uint8)
        chili.mask_seed[2, :] = 255
        chili.mask_endocarp = np.zeros((10, 10), dtype=np.uint8)
        chili.mask_endocarp[3, :] = 255
        chili.mask_placenta = np.zeros((10, 10), dtype=np.uint8)

        chili.mask_placenta[1, :9] = 255
        self.assertIsNone(chili.get_flesh_placenta_ratio())

        chili.mask_placenta[1, 9] = 255
        self.assertEqual(chili.get_flesh_placenta_ratio(), 7.0)
        self.assertTrue(
            bool(np.all(chili._get_flesh_mask()[chili.mask_endocarp > 0] > 0))
        )


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
    def synthetic_mesocarp(angle_deg=0.0):
        size = 121
        yy, xx = np.mgrid[:size, :size]
        angle = np.deg2rad(angle_deg)
        normal = np.array([-np.sin(angle), np.cos(angle)])
        signed_normal = (xx - 60) * normal[0] + (yy - 60) * normal[1]
        mesocarp = np.abs(signed_normal) <= 5.0
        inner_reference = signed_normal < -5.0
        outer_reference = signed_normal > 5.0
        return {
            "mesocarp": mesocarp,
            "distance_to_inner": ndi.distance_transform_edt(~inner_reference),
            "distance_to_outer": ndi.distance_transform_edt(~outer_reference),
            "signed_distance": ndi.distance_transform_edt(mesocarp)
            - ndi.distance_transform_edt(~mesocarp),
        }

    def test_straight_and_diagonal_mesocarp(self):
        for angle, expected_px in ((0.0, 11.0), (27.0, 10.0)):
            with self.subTest(angle=angle):
                result = measure_skeleton_point(
                    (60.0, 60.0), self.synthetic_mesocarp(angle)
                )
                self.assertIsNotNone(result)
                self.assertLessEqual(abs(result["length_px"] - expected_px), 0.3)

    def test_mesocarp_thickness_schema_is_ordered(self):
        measured = pd.DataFrame(
            {
                "measurement_status": ["measured"] * 3,
                "thickness_mm": [2.0, 4.0, 6.0],
            }
        )
        with patch(
            "src.geometric_features.measure_mesocarp_thickness",
            return_value=measured,
        ), patch(
            "src.geometric_features.filter_clean_thickness_points",
            return_value=measured,
        ):
            stats = calculate_mesocarp_thickness_stats({}, ratio=0.1)

        self.assertEqual(
            set(stats),
            {
                "min_mesocarp_thickness",
                "avg_mesocarp_thickness",
                "max_mesocarp_thickness",
            },
        )
        self.assertLessEqual(
            stats["min_mesocarp_thickness"], stats["avg_mesocarp_thickness"]
        )
        self.assertLessEqual(
            stats["avg_mesocarp_thickness"], stats["max_mesocarp_thickness"]
        )

    def test_curved_annulus_crosses_both_edges(self):
        yy, xx = np.mgrid[:151, :151]
        radius = np.hypot(xx - 75, yy - 75)
        mesocarp = (radius >= 35) & (radius <= 45)
        data = {
            "mesocarp": mesocarp,
            "distance_to_inner": ndi.distance_transform_edt(~(radius < 35)),
            "distance_to_outer": ndi.distance_transform_edt(~(radius > 45)),
            "signed_distance": ndi.distance_transform_edt(mesocarp)
            - ndi.distance_transform_edt(~mesocarp),
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


class FruitWidthTests(unittest.TestCase):
    def _chili(self, mask):
        chili = Chili.__new__(Chili)
        chili.mask_chili = mask
        chili.ratio = 0.1
        return chili

    def test_nine_sections_use_eight_interior_width_chords(self):
        mask = np.zeros((101, 80), dtype=np.uint8)
        mask[10:91, 20:61] = 255
        chili = self._chili(mask)

        sections = chili._measure_fruit_sections()
        stats = chili.get_width_fruit()
        width_9seg, length_9seg = chili.get_width_length_fruit_9seg()

        self.assertEqual(len(sections), 10)
        self.assertEqual([s["width_px"] for s in sections[1:-1]], [40.0] * 8)
        self.assertEqual(stats, {"min": 4.0, "max": 4.0, "avg": 4.0})
        self.assertEqual(width_9seg, stats["avg"])
        self.assertEqual(length_9seg, 8.0)

    def test_empty_or_too_short_mask_returns_zero_widths(self):
        empty = self._chili(np.zeros((20, 20), dtype=np.uint8))
        short_mask = np.zeros((20, 20), dtype=np.uint8)
        short_mask[5:10, 5:15] = 255
        short = self._chili(short_mask)

        self.assertEqual(empty.get_width_fruit(), {"min": 0, "max": 0, "avg": 0})
        self.assertEqual(empty.get_width_length_fruit_9seg(), (0.0, 0.0))
        self.assertEqual(short.get_width_fruit(), {"min": 0, "max": 0, "avg": 0})
        self.assertEqual(short.get_width_length_fruit_9seg(), (0.0, 0.0))

    def test_missing_interior_scanline_does_not_return_partial_width(self):
        mask = np.zeros((101, 80), dtype=np.uint8)
        mask[10:91, 20:61] = 255
        mask[46, :] = 0
        chili = self._chili(mask)

        self.assertEqual(chili.get_width_fruit(), {"min": 0, "max": 0, "avg": 0})
        self.assertEqual(chili.get_width_length_fruit_9seg(), (0.0, 0.0))


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

    def test_wrinkle_uses_chili_minus_pedicel_and_outer(self):
        chili = Chili.__new__(Chili)
        shape = (80, 100)
        chili.mask_chili = np.zeros(shape, dtype=np.uint8)
        chili.mask_chili[10:70, 10:90] = 255
        chili.mask_pedicel = np.zeros(shape, dtype=np.uint8)
        chili.mask_pedicel[:25, 40:61] = 255
        chili.mask_outer = np.zeros(shape, dtype=np.uint8)
        chili.mask_outer[:, 80:] = 255
        chili.mask_endocarp = np.zeros(shape, dtype=np.uint8)
        chili.mask_endocarp[30:60, 20:70] = 255
        chili.mask_pcp = chili.mask_pedicel.copy()
        chili.mask_seed = np.zeros(shape, dtype=np.uint8)
        chili.mask_mesocarp = (
            (chili.mask_chili > 0)
            & ~(chili.mask_pedicel > 0)
            & ~(chili.mask_outer > 0)
            & ~(chili.mask_endocarp > 0)
        ).astype(np.uint8) * 255
        data = build_mesocarp_data(chili, 0.1)
        expected = (chili.mask_chili > 0) & ~(
            (chili.mask_pedicel > 0) | (chili.mask_outer > 0)
        )
        np.testing.assert_array_equal(data["wrinkle_mask"], expected)
        runs = extract_chili_body_contours(data)
        self.assertEqual(len(runs), 1)
        xy = np.rint(runs[0]).astype(int)
        self.assertTrue(expected[xy[:, 1], xy[:, 0]].all())
        self.assertEqual(xy[:, 0].max(), 79)
        # The boundary exposed by removing the pedicel is included.
        self.assertTrue(np.any((xy[:, 1] == 25) & (xy[:, 0] == 50)))
        np.testing.assert_array_equal(runs[0][0], runs[0][-1])
        # Internal tissues and mesocarp geometry do not affect extraction.
        data["mesocarp"][:] = False
        data["distance_to_inner"][:] = 0
        np.testing.assert_array_equal(extract_chili_body_contours(data)[0], runs[0])

    def test_wrinkle_ignores_holes_and_keeps_external_components(self):
        body = np.zeros((100, 100), dtype=bool)
        body[10:60, 10:60] = True
        body[20:40, 20:40] = False
        body[70:90, 70:90] = True
        runs = extract_chili_body_contours({"wrinkle_mask": body})
        self.assertEqual(len(runs), 2)
        for run in runs:
            xy = np.rint(run).astype(int)
            self.assertFalse(np.any(
                (xy[:, 0] >= 19) & (xy[:, 0] <= 40)
                & (xy[:, 1] >= 19) & (xy[:, 1] <= 40)
            ))

    def test_closed_loop_preserves_full_chord_cuts_without_degenerate_tail(self):
        angles = np.linspace(0, 2 * np.pi, 361)
        contour = np.column_stack([20 * np.cos(angles), 20 * np.sin(angles)])
        # Tail balancing on this closed loop previously erased a valid cut.
        segments = segment_polyline_by_exact_chord(contour, 35.0)[2]
        self.assertEqual(len(segments), 3)
        self.assertAlmostEqual(sum(s["arc_px"] for s in segments),
                               np.linalg.norm(np.diff(contour, axis=0), axis=1).sum())
        for segment in segments[:-1]:
            self.assertAlmostEqual(segment["chord_px"], 35.0, places=5)
        for segment in segments:
            self.assertGreater(segment["chord_px"], 1.0)
            self.assertLessEqual(segment["chord_px"], 35.0 + 1e-6)
            self.assertLess(segment["curvature_excess"], 3.0)

    def test_closed_loop_smaller_than_threshold_has_valid_segments(self):
        angles = np.linspace(0, 2 * np.pi, 361)
        contour = np.column_stack([5 * np.cos(angles), 5 * np.sin(angles)])
        segments = segment_polyline_by_exact_chord(contour, 50.0)[2]
        self.assertEqual(len(segments), 3)
        for segment in segments:
            self.assertGreater(segment["chord_px"], 1.0)
            self.assertLessEqual(segment["chord_px"], 50.0)
            self.assertLess(segment["curvature_excess"], 1.0)


if __name__ == "__main__":
    unittest.main()
