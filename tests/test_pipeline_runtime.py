import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image

from src.pipeline import PipelineConfig, create_manifest, feature_row, feature_columns, write_exports
from src.object import Chili, Fruits
from src.geometric_features import calculate_mesocarp_thickness_stats, calculate_mesocarp_wrinkle_index
from predictor import parse_args, config_from_args


class RuntimeTests(unittest.TestCase):
    def test_exif_image_metadata_uses_runtime_decoded_frame(self):
        import cv2
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'rotated.jpg'
            exif=Image.Exif();exif[274]=6
            Image.new('RGB',(20,40),'black').save(path,exif=exif)
            fruits=Fruits(str(path),model_detect=SimpleNamespace(predict=lambda p:[]),model_segment=None)
            decoded=cv2.imread(str(path))
            self.assertEqual((fruits.width_img,fruits.height_img),(decoded.shape[1],decoded.shape[0]))
            self.assertEqual(fruits.stored_image_size,(20,40))
            fruits.image.close()

    def test_cli_config_preserves_effective_overrides_and_aliases(self):
        args = parse_args(['--input_path','example.png','--name_model_det','none', '--seg-imgsz','960', '--det-crop-top','0', '--det-conf-thres','0.42', '--chili_threshold','23', '--mesocarp-wrinkle-segment-threshold-mm','10'])
        config=config_from_args(args)
        self.assertEqual(config.detector_backend,'threshold')
        self.assertEqual((config.seg_imgsz,config.det_crop_top,config.chili_threshold,config.wrinkle_tau_mm),(960,0,23,10))
        self.assertEqual(config.effective_settings()['detector_threshold_offset'],22)
        self.assertEqual(config.det_conf,.42)

    def test_rejects_nonfinite_and_invalid_configuration_before_model_loading(self):
        for kwargs in ({'wrinkle_tau_mm':float('nan')}, {'det_conf':float('inf')}, {'seg_imgsz':0}, {'det_crop_top':-1}, {'seg_imgsz':12.5}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): PipelineConfig(**kwargs)

    def test_web_cli_share_values_and_keep_existing_column_schemas(self):
        c=SimpleNamespace(id=2,name='sample_2.jpg',length_fruit=10,width_fruit=4,
            width_fruit_stats={'min':2,'max':6},area_fruit=30,mesocarp_area=12,
            mesocarp_thickness_stats={'min_mesocarp_thickness':1,'avg_mesocarp_thickness':2,'max_mesocarp_thickness':3},
            placenta_area=5,seed_stats={'sum':7,'min':1,'max':3,'avg':2,'std':.5},number_seeds=4,
            mesocarp_area_fraction=.6,seed_density=.25,curvature_index=.9,asymmetry_index=.1,mesocarp_wrinkle_index=.05,
            width_bbox=4,height_bbox=10,width_fruit_9seg=4,length_fruit_9seg=10,fruit_body_area=20,
            redness=30,color_uniformity=5,ratio=.1,scale_source='qr')
        web=feature_row(c);cli=feature_row(c,'sample');cli.pop('Name Root')
        self.assertEqual(web,cli)
        self.assertEqual(len(feature_columns()),32)
        self.assertEqual(len(feature_columns(True)),33)

    def test_exports_create_directory_and_manifest_tracks_input_and_weights(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);weight=root/'model.pt';weight.write_bytes(b'checkpoint')
            image=root/'sample.png';Image.new('RGB',(8,8)).save(image)
            manifest=create_manifest(PipelineConfig(detector_backend='threshold',weight_seg=str(weight)),[image])
            self.assertEqual(manifest['configuration']['seg_imgsz'],640)
            self.assertEqual(len(manifest['checkpoints']['segmentation']['sha256']),64)
            self.assertEqual(manifest['inputs'][0]['size_bytes'],image.stat().st_size)
            csv,meta=write_exports(root/'new/subdir',pd.DataFrame([{'ID':1}]),manifest)
            self.assertEqual(csv.read_text(),'ID\n1\n')
            self.assertEqual(json.loads(meta.read_text())['inputs'],manifest['inputs'])
            self.assertTrue(manifest['source_files'])

    def test_thickness_diagnostic_records_prefilter_fallback_without_changing_stats(self):
        measured=pd.DataFrame({'measurement_status':['measured']*3,'thickness_mm':[1.,2.,3.]})
        diagnostics={}
        with patch('src.geometric_features.measure_mesocarp_thickness',return_value=measured),patch('src.geometric_features.filter_clean_thickness_points',return_value=measured.iloc[:1]):
            plain=calculate_mesocarp_thickness_stats({},.1)
            observed=calculate_mesocarp_thickness_stats({},.1,diagnostics)
        self.assertEqual(observed,plain)
        self.assertEqual((diagnostics['valid_count'],diagnostics['clean_count'],diagnostics['used_count']),(3,1,3))
        self.assertTrue(diagnostics['used_prefilter_samples'])

    def test_empty_wrinkle_has_unavailable_status(self):
        diagnostics={}
        value=calculate_mesocarp_wrinkle_index({'wrinkle_mask':np.zeros((20,20),bool)},.1,diagnostics=diagnostics)
        self.assertIsNone(value)
        self.assertEqual(diagnostics['segment_count'],0)
        self.assertEqual(diagnostics['status'],'unavailable')

    def test_failed_pedicel_keeps_legacy_masks_and_records_failure(self):
        c=Chili.__new__(Chili);c.mask_pcp=np.ones((10,10),np.uint8)*255
        failed=(None,None,None,None,None,None,False,'failed')
        with patch('src.object.run_case1_armpit',return_value=failed),patch('src.object.run_case2_geo_original',return_value=failed):
            stem,placenta=c._separate_pedicel_and_placenta()
        self.assertFalse(np.any(stem));np.testing.assert_array_equal(placenta,c.mask_pcp)
        self.assertEqual(c._pedicel_diagnostics['status'],'failed')

if __name__=='__main__': unittest.main()
