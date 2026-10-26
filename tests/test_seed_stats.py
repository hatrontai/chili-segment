import unittest
from unittest.mock import patch
import numpy as np
from src.object import Chili


class SeedStatsTests(unittest.TestCase):
    def make_chili(self, masks):
        obj = Chili.__new__(Chili)
        obj.ratio = 0.5
        obj.rseg = {'masks_data': [{'class_id': 0, 'mask': m, 'source': 'yolo'} for m in masks]}
        obj.mask_seed = np.any(masks, axis=0).astype(np.uint8) if masks else np.zeros((12, 12), np.uint8)
        return obj

    def test_overlapping_and_small_instances_are_preserved(self):
        a = np.zeros((12, 12), np.uint8); a[2:6, 2:6] = 1
        b = np.zeros_like(a); b[2:6, 4:8] = 1
        small = np.zeros_like(a); small[9, 9] = 1
        obj = self.make_chili([a, b, small, np.zeros_like(a)])
        with patch.object(obj, '_get_seed_stats_watershed', side_effect=AssertionError('unexpected fallback')):
            stats = obj.get_seed_stats()
        # Two 16-pixel seeds share 8 pixels; the one-pixel seed is retained.
        self.assertEqual(stats, {'count': 3, 'sum': 6.25, 'min': .25, 'max': 4., 'avg': 2.75, 'std': 1.77})

    def test_legacy_union_does_not_invent_yolo_instances(self):
        obj = self.make_chili([])
        obj.mask_seed[2:5, 2:5] = 1
        obj.mask_seed[8:11, 8:11] = 1
        with patch.object(obj, '_get_seed_stats_watershed', side_effect=AssertionError('unexpected fallback')):
            self.assertEqual(obj.get_seed_stats(), dict(sum=0, min=0, max=0, avg=0, std=0, count=0))

    def test_identical_masks_count_as_two_instances_but_one_area(self):
        mask = np.zeros((12, 12), np.uint8)
        mask[2:6, 2:6] = 255
        obj = self.make_chili([mask, mask.copy()])
        self.assertEqual(obj.get_seed_stats()['count'], 2)
        self.assertEqual(obj.get_seed_stats()['sum'], 4.0)

    def test_other_classes_and_non_yolo_masks_are_excluded(self):
        mask = np.zeros((12, 12), np.uint8)
        mask[2:4, 2:4] = 255
        obj = self.make_chili([mask])
        obj.rseg['masks_data'].extend([
            {'class_id': 1, 'mask': np.ones_like(mask), 'source': 'yolo'},
            {'class_id': 0, 'mask': np.ones_like(mask), 'source': 'synthetic'},
        ])
        self.assertEqual(obj.get_seed_stats()['count'], 1)
        self.assertEqual(obj.get_seed_stats()['sum'], 1.0)

    def test_explicit_watershed_keeps_previous_small_region_filter(self):
        a = np.zeros((12, 12), np.uint8); a[2:6, 2:6] = 1
        small = np.zeros_like(a); small[9, 9] = 1
        obj = self.make_chili([a, small])
        self.assertEqual(obj.get_seed_stats()['count'], 2)
        self.assertEqual(obj.get_seed_stats(method='watershed')['count'], 1)

    def test_no_seed_and_invalid_method(self):
        obj = self.make_chili([])
        self.assertEqual(obj.get_seed_stats(), dict(sum=0, min=0, max=0, avg=0, std=0, count=0))
        with self.assertRaises(ValueError):
            obj.get_seed_stats(method='unknown')


if __name__ == '__main__':
    unittest.main()
