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


class SignedDeviationTests(unittest.TestCase):
    def make_chili(self, centers):
        chili = Chili.__new__(Chili)
        chili.mask_fruit = np.zeros((len(centers), 100), dtype=np.uint8)
        for y, center in enumerate(centers):
            chili.mask_fruit[y, center - 10:center + 11] = 255
        chili.mask_outer = np.zeros_like(chili.mask_fruit)
        chili.mask_pedicel = np.zeros_like(chili.mask_fruit)
        chili.spine = chili._calculate_spine()
        return chili

    def test_straight_and_tilted_centerlines_have_zero_deviation(self):
        for centers in ([30] * 11, [20 + y for y in range(11)]):
            self.assertEqual(self.make_chili(centers).get_deviation_stats(),
                             {"D_L": 0.0, "D_R": 0.0, "D_mean": 0.0})

    def test_bowed_body_and_mirror_reverse_signs(self):
        chili = self.make_chili([20 + y * (10 - y) for y in range(11)])
        with patch.object(chili, '_calculate_spine', side_effect=AssertionError('must reuse FL spine')):
            right = chili.get_deviation_stats()
        self.assertEqual(right['D_L'], 0)
        self.assertGreater(right['D_R'], 0)
        self.assertGreater(right['D_mean'], 0)
        chili.mask_fruit = np.fliplr(chili.mask_fruit).copy()
        chili.spine = chili._calculate_spine()
        left = chili.get_deviation_stats()
        self.assertAlmostEqual(left['D_L'], -right['D_R'], places=4)
        self.assertEqual(left['D_R'], 0)
        self.assertAlmostEqual(left['D_mean'], -right['D_mean'], places=4)

    def test_equal_row_weight_zero_width_and_epsilon(self):
        chili = self.make_chili([30] * 5)
        # Fixed shared spine at x=30; one narrow row is wholly on the right.
        chili.mask_fruit[:] = 0
        chili.mask_fruit[0, 20:41] = 255  # centered, width 20
        chili.mask_fruit[1, 31:33] = 255  # right, width 1
        chili.mask_fruit[2, 30] = 255    # zero width still counts in the mean
        chili.mask_fruit[4, 20:41] = 255
        chili.spine['valid'][3] = False  # an empty row is excluded
        stats = chili.get_deviation_stats()
        self.assertEqual(stats, {'D_L': 0.0, 'D_R': 1.0, 'D_mean': 0.25})
        self.assertTrue(np.isfinite(list(stats.values())).all())

    def test_opposite_deviations_cancel_only_in_mean(self):
        chili = self.make_chili([30] * 4)
        chili.mask_fruit[:] = 0
        chili.mask_fruit[0, 10:20] = 255
        chili.mask_fruit[1, 40:50] = 255
        chili.mask_fruit[2:, 20:41] = 255
        stats = chili.get_deviation_stats()
        self.assertLess(stats['D_L'], 0)
        self.assertGreater(stats['D_R'], 0)
        self.assertEqual(stats['D_mean'], 0)

    def test_pedicel_and_outer_are_excluded(self):
        chili = self.make_chili([30] * 11)
        before = chili.get_deviation_stats()
        chili.mask_fruit[:, 60:70] = 255
        chili.mask_pedicel[:, 60:65] = 255
        chili.mask_outer[:, 65:70] = 255
        chili.spine = chili._calculate_spine()
        self.assertEqual(chili.get_deviation_stats(), before)

    def test_empty_or_single_row_region_returns_zeros(self):
        chili = self.make_chili([30])
        self.assertEqual(chili.get_deviation_stats(), {'D_L': 0., 'D_R': 0., 'D_mean': 0.})
        chili.mask_fruit[:] = 0
        chili.spine = chili._calculate_spine()
        self.assertEqual(chili.get_deviation_stats(), {'D_L': 0., 'D_R': 0., 'D_mean': 0.})


class _NoDetectionModel:
    def predict(self, _path):
        return []


class FruitBodyLengthTests(unittest.TestCase):
    def make_chili(self):
        chili = Chili.__new__(Chili)
        chili.ratio = 0.1
        chili.mask_fruit = np.zeros((100, 80), dtype=np.uint8)
        chili.mask_fruit[20:80, 20:60] = 255
        chili.mask_fruit[5:20, 20:25] = 255
        chili.mask_fruit[80:90, 20:60] = 255
        chili.mask_pedicel = np.zeros_like(chili.mask_fruit)
        chili.mask_pedicel[5:20, 20:25] = 255
        chili.mask_outer = np.zeros_like(chili.mask_fruit)
        chili.mask_outer[80:90, 20:60] = 255
        return chili

    def test_length_excludes_pedicel_and_outer_and_curvature_uses_body_endpoints(self):
        chili = self.make_chili()
        chili.spine = chili._calculate_spine()
        self.assertEqual((chili.spine['y_min'], chili.spine['y_max']), (20, 79))
        chili.length_fruit = chili.get_length_fruit()
        self.assertEqual(chili.length_fruit, 6.0)
        self.assertAlmostEqual(chili.get_curvature_index(), 1 - (5.9 + 1e-6) / (6.0 + 1e-6), places=4)
        # Width keeps its previous measurement protocol.
        widths = chili.get_width_fruit()
        chili.mask_pedicel[:] = 0
        chili.spine = chili._calculate_spine()
        self.assertEqual(chili.spine['y_min'], 5)
        self.assertGreater(chili.get_length_fruit(), 6.0)
        self.assertEqual(chili.get_width_fruit(), widths)

    def test_fruit_area_is_body_area_and_exclusions_are_not_double_counted(self):
        chili = self.make_chili()
        self.assertEqual(chili.get_area(), 24.0)
        self.assertEqual(chili.get_area(), chili.get_fruit_body_area())
        # Overlapping exclusion masks must be subtracted as a union.
        chili.mask_pedicel |= chili.mask_outer
        self.assertEqual(chili.get_area(), 24.0)
        chili.mask_pedicel = chili.mask_fruit.copy()
        self.assertEqual(chili.get_area(), 0.0)

    def test_empty_body_has_no_length_spine(self):
        chili = self.make_chili()
        chili.mask_pedicel = chili.mask_fruit.copy()
        chili.spine = chili._calculate_spine()
        self.assertIsNone(chili.spine)
        self.assertEqual(chili.get_length_fruit(), 0)


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
            "MAF", "SCR", "FCI", "D_L", "D_R", "D_mean", "MWI",
        ]
        self.assertEqual([item.symbol for item in CANONICAL_TRAITS], expected)
        self.assertEqual(len(set(expected)), 19)
        units = {item.symbol: item.unit for item in CANONICAL_TRAITS}
        self.assertEqual(units["FL"], "mm")
        self.assertEqual(units["FA"], "mm^2")
        self.assertEqual(units["SC"], "seed")
        self.assertEqual(units["FCI"], "dimensionless")
        self.assertEqual(len(CANONICAL_TRAIT_COLUMNS), 19)

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
        chili.deviation_stats = {"D_L": -0.2, "D_R": 0.4, "D_mean": 0.1}
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
        self.assertEqual({k: values[k] for k in chili.deviation_stats}, chili.deviation_stats)
        self.assertEqual(list(row), list(CANONICAL_TRAIT_COLUMNS))

    def test_maf_excludes_outer_and_pedicel_and_is_scale_independent(self):
        chili = Chili.__new__(Chili)
        chili.mask_fruit = np.full((10, 10), 255, dtype=np.uint8)
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
        chili.mask_fruit[:] = 0
        chili.mask_mesocarp[:] = 0
        self.assertIsNone(chili.get_mesocarp_area_fraction())

    def test_scr_uses_current_endocarp_as_cavity(self):
        chili = Chili.__new__(Chili)
        chili.mask_seed = np.zeros((4, 4), dtype=np.uint8)
        chili.mask_endocarp = np.zeros((4, 4), dtype=np.uint8)
        chili.mask_placenta = np.zeros((4, 4), dtype=np.uint8)
        chili.mask_seed[0, :2] = 255
        chili.mask_endocarp[:2, :] = 255
        chili.mask_placenta[2, :2] = 255
        self.assertEqual(chili.get_seed_density(), 0.25)
        # Seed pixels outside endocarp must not enlarge the denominator.
        chili.mask_seed[3, :2] = 255
        self.assertEqual(chili.get_seed_density(), 0.5)
        # Independent predictions can yield a ratio above one; do not clamp.
        chili.mask_seed[:] = 255
        self.assertEqual(chili.get_seed_density(), 2.0)
        chili.mask_endocarp[:] = 0
        self.assertEqual(chili.get_seed_density(), 0)

    def test_fci_is_regularized_chord_complement_and_zero_means_straight(self):
        chili = Chili.__new__(Chili)
        chili.ratio = 0.1
        chili.spine = {
            "coeffs": np.array([0.0, 0.0, 0.0, 0.0]),
            "y_min": 0,
            "y_max": 10,
        }
        chili.length_fruit = 1.0
        self.assertEqual(chili.get_curvature_index(), 0.0)
        chili.length_fruit = 2.0
        self.assertEqual(chili.get_curvature_index(), 0.5)
        chili.length_fruit = 4.0
        self.assertEqual(chili.get_curvature_index(), 0.75)
        # At tiny scales the epsilon must occur in both numerator and denominator.
        chili.ratio = 1e-7
        chili.length_fruit = 3e-6
        self.assertEqual(chili.get_curvature_index(), 0.5)
        chili.length_fruit = 0.0
        self.assertEqual(chili.get_curvature_index(), 0.0)
        chili.spine = None
        self.assertEqual(chili.get_curvature_index(), 0.0)


    def test_mesocarp_excludes_all_non_mesocarp_classes(self):
        shape = (60, 80)
        chili = type("SyntheticChili", (), {})()
        chili.mask_fruit = np.full(shape, 255, dtype=np.uint8)
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

        self.assertFalse(bool(np.any(data["raw_mesocarp"] & excluded)))

    def test_placenta_area_counts_full_mask_including_outer_and_seed_overlap(self):
        chili = Chili.__new__(Chili)
        chili.ratio = 0.2
        chili.mask_placenta = np.zeros((10,10), dtype=np.uint8)
        chili.mask_placenta[2:7, 3:7] = 255
        chili.mask_outer = chili.mask_placenta.copy()
        chili.mask_seed = chili.mask_placenta.copy()
        self.assertEqual(chili.get_placenta_area(), 0.8)
        chili.mask_placenta[:] = 0
        self.assertEqual(chili.get_placenta_area(), 0.0)

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
        chili.mask_fruit = np.full((10, 10), 255, dtype=np.uint8)
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

    def test_smoothed_mask_controls_both_boundaries_and_skeleton(self):
        from types import SimpleNamespace
        raw = np.zeros((101,101), dtype=np.uint8)
        raw[40:61, 10:91] = 255
        smooth = np.zeros_like(raw, dtype=bool)
        smooth[45:56, 10:91] = True
        zero = np.zeros_like(raw)
        endocarp = zero.copy(); endocarp[:40, :] = 255
        fruit = zero.copy(); fruit[:61, :] = 255
        chili = SimpleNamespace(mask_fruit=fruit, mask_endocarp=endocarp,
                                mask_outer=zero, mask_pcp=zero, mask_seed=zero,
                                mask_pedicel=zero, mask_mesocarp=raw)
        with patch('src.geometric_features.smooth_mesocarp_for_skeleton',return_value=smooth):
            data = build_mesocarp_data(chili,.1)
        np.testing.assert_array_equal(data['mesocarp'], smooth)
        self.assertFalse(np.any(data['measurement_skeleton'] & ~smooth))
        self.assertFalse(np.any(data['boundary'] & ~smooth))
        self.assertLess(data['signed_distance'][42,50],0)
        result = measure_skeleton_point((50.,50.),data)
        self.assertAlmostEqual(result['length_px'],11.,places=2)
        np.testing.assert_array_equal(chili.mask_mesocarp,raw)
        # An erased wall stays unavailable instead of reverting to the raw mask.
        with patch('src.geometric_features.smooth_mesocarp_for_skeleton',return_value=np.zeros_like(smooth)):
            empty = build_mesocarp_data(chili,.1)
        self.assertFalse(empty['measurement_skeleton'].any())
        self.assertIsNone(calculate_mesocarp_thickness_stats(empty,.1)['avg_mesocarp_thickness'])

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
        chili.mask_fruit = mask
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
    def test_segment_score_uses_chord_over_arc_complement(self):
        # A 3-4-5 bend: arc 7 and chord 5, not the old 7/5 - 1.
        points = np.array([[0.,0.],[3.,0.],[3.,4.]])
        segments = segment_polyline_by_exact_chord(points, 10.)[2]
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]['wrinkle_score'], 2/7)
        self.assertGreaterEqual(segments[0]['wrinkle_score'], 0)
        self.assertLess(segments[0]['wrinkle_score'], 1)

    def test_runtime_averages_segment_scores_with_equal_weights(self):
        from src.geometric_features import calculate_mesocarp_wrinkle_index
        segments = [{'wrinkle_score': 0.0}, {'wrinkle_score': 0.375}]
        diagnostics = {}
        with patch('src.geometric_features.extract_chili_body_contours', return_value=[np.zeros((3,2))]), patch('src.geometric_features.segment_polyline_by_exact_chord', return_value=(None,None,segments)):
            value = calculate_mesocarp_wrinkle_index({}, .1, diagnostics=diagnostics)
        self.assertEqual(value, 0.1875)
        self.assertEqual(diagnostics['maximum_segment_wrinkle'], .375)

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
            [segment["wrinkle_score"] for segment in straight_segments]
        )
        zigzag_wrinkle = np.mean(
            [segment["wrinkle_score"] for segment in zigzag_segments]
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
        chili.mask_fruit = np.zeros(shape, dtype=np.uint8)
        chili.mask_fruit[10:70, 10:90] = 255
        chili.mask_pedicel = np.zeros(shape, dtype=np.uint8)
        chili.mask_pedicel[:25, 40:61] = 255
        chili.mask_outer = np.zeros(shape, dtype=np.uint8)
        chili.mask_outer[:, 80:] = 255
        chili.mask_endocarp = np.zeros(shape, dtype=np.uint8)
        chili.mask_endocarp[30:60, 20:70] = 255
        chili.mask_pcp = chili.mask_pedicel.copy()
        chili.mask_seed = np.zeros(shape, dtype=np.uint8)
        chili.mask_mesocarp = (
            (chili.mask_fruit > 0)
            & ~(chili.mask_pedicel > 0)
            & ~(chili.mask_outer > 0)
            & ~(chili.mask_endocarp > 0)
        ).astype(np.uint8) * 255
        data = build_mesocarp_data(chili, 0.1)
        expected = (chili.mask_fruit > 0) & ~(
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
            self.assertLess(segment["wrinkle_score"], 3.0)

    def test_closed_loop_smaller_than_threshold_has_valid_segments(self):
        angles = np.linspace(0, 2 * np.pi, 361)
        contour = np.column_stack([5 * np.cos(angles), 5 * np.sin(angles)])
        segments = segment_polyline_by_exact_chord(contour, 50.0)[2]
        self.assertEqual(len(segments), 3)
        for segment in segments:
            self.assertGreater(segment["chord_px"], 1.0)
            self.assertLessEqual(segment["chord_px"], 50.0)
            self.assertLess(segment["wrinkle_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
