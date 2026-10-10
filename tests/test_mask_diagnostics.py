"""Coordinate/matching tests for diagnostic reporting; no physical validation."""
import unittest
import numpy as np
from evaluate_measurement_masks import Instance, match_instances, pair_metrics, restore

class MaskDiagnosticTests(unittest.TestCase):
    def test_identical_masks_have_unit_scores(self):
        a=Instance(10,20,np.ones((6,8),bool))
        self.assertEqual(pair_metrics(a,a),{'iou':1.,'dice':1.,'boundary_f':1.})

    def test_matching_is_one_to_one_and_counts_missing_and_extra(self):
        a=Instance(10,20,np.ones((6,8),bool));b=Instance(30,20,np.ones((6,8),bool))
        r=match_instances([a,a],[a,b])
        self.assertEqual((r['matched'],r['false_positive'],r['false_negative']),(1,1,1))

    def test_nonoverlapping_global_coordinates_do_not_match(self):
        mask=np.ones((6,8),bool)
        r=match_instances([Instance(0,0,mask)],[Instance(1000,1000,mask)])
        self.assertEqual(r['matched'],0)

    def test_inverse_rotation_restores_source_mask(self):
        import cv2
        mask=np.zeros((10,20),np.uint8);mask[2:5,3:9]=255
        oriented=cv2.rotate(mask,cv2.ROTATE_90_CLOCKWISE)
        np.testing.assert_array_equal(restore(oriented,'90_cw'),mask)

if __name__=='__main__':unittest.main()
