"""Small diagnostic of deployed threshold and optional YOLO chili masks.

Uses only bundled annotated images. This is not a held-out paper experiment.
Masks are projected from oriented crops into original image coordinates.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt, binary_erosion
from src.pipeline import PipelineConfig, build_models, run_image, create_manifest, image_diagnostics

@dataclass
class Instance:
    x: int
    y: int
    mask: np.ndarray


def common_masks(a,b):
    x=min(a.x,b.x);y=min(a.y,b.y)
    right=max(a.x+a.mask.shape[1],b.x+b.mask.shape[1]);bottom=max(a.y+a.mask.shape[0],b.y+b.mask.shape[0])
    aa=np.zeros((bottom-y,right-x),bool);bb=np.zeros_like(aa)
    aa[a.y-y:a.y-y+a.mask.shape[0],a.x-x:a.x-x+a.mask.shape[1]]=a.mask
    bb[b.y-y:b.y-y+b.mask.shape[0],b.x-x:b.x-x+b.mask.shape[1]]=b.mask
    return aa,bb


def overlap(a,b):
    # Avoid allocating a large union canvas for nonoverlapping candidates.
    if a.x+a.mask.shape[1] <= b.x or b.x+b.mask.shape[1] <= a.x or a.y+a.mask.shape[0] <= b.y or b.y+b.mask.shape[0] <= a.y:
        return 0.
    aa,bb=common_masks(a,b)
    union=np.count_nonzero(aa|bb)
    return np.count_nonzero(aa&bb)/union if union else 0.


def pair_metrics(a,b,tolerance=2):
    aa,bb=common_masks(a,b);intersection=np.count_nonzero(aa&bb)
    iou=intersection/np.count_nonzero(aa|bb)
    dice=2*intersection/(np.count_nonzero(aa)+np.count_nonzero(bb))
    # 8-neighbour boundary, with a declared Euclidean tolerance in ORIGINAL pixels.
    ae=aa & ~binary_erosion(aa,structure=np.ones((3,3)),border_value=0)
    be=bb & ~binary_erosion(bb,structure=np.ones((3,3)),border_value=0)
    precision=np.mean(distance_transform_edt(~be)[ae]<=tolerance) if np.any(ae) else 0.
    recall=np.mean(distance_transform_edt(~ae)[be]<=tolerance) if np.any(be) else 0.
    f=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {'iou':float(iou),'dice':float(dice),'boundary_f':float(f)}


def match_instances(predictions,truth,threshold=.5,tolerance=2):
    matrix=np.array([[overlap(p,g) for g in truth] for p in predictions],float).reshape(len(predictions),len(truth))
    pairs=[]
    if matrix.size:
        # Maximize qualifying match count first, then overlap.
        reward=(matrix>=threshold)*(1+min(matrix.shape))+matrix
        rows,cols=linear_sum_assignment(-reward)
        pairs=[(int(r),int(c)) for r,c in zip(rows,cols) if matrix[r,c]>=threshold]
    return {'matched':len(pairs),'false_positive':len(predictions)-len(pairs),
            'false_negative':len(truth)-len(pairs),'pairs':[
                {'prediction':r,'ground_truth':c,**pair_metrics(predictions[r],truth[c],tolerance)} for r,c in pairs]}


def ground_truth(path,width,height):
    data=json.loads(path.read_text());instances=[]
    for shape in data['shapes']:
        if shape['label'].strip().lower()!='chili':continue
        points=np.rint(shape['points']).astype(np.int32)
        x,y,w,h=cv2.boundingRect(points);x1=max(0,x);y1=max(0,y);x2=min(width,x+w);y2=min(height,y+h)
        if x2<=x1 or y2<=y1:continue
        mask=np.zeros((y2-y1,x2-x1),np.uint8);cv2.fillPoly(mask,[points-[x1,y1]],1)
        instances.append(Instance(x1,y1,mask>0))
    return instances


def restore(mask,rotation):
    if rotation=='90_cw':return cv2.rotate(mask,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation=='90_ccw':return cv2.rotate(mask,cv2.ROTATE_90_CLOCKWISE)
    return mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--annotation-dir',type=Path,default=Path(__file__).resolve().parents[2]/'jeju_segmentation',help='Current five-class LabelMe annotations; bundled JSON files are historical')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seg-imgsz',type=int,default=640)
    args=parser.parse_args()
    paths=sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'} and (args.annotation_dir/p.with_suffix('.json').name).exists())
    config=PipelineConfig(seg_imgsz=args.seg_imgsz)
    manifest=create_manifest(config,paths);models=build_models(config)
    from src.pipeline import file_identity
    manifest['annotations']=[file_identity(args.annotation_dir/p.with_suffix('.json').name) for p in paths]
    report={'purpose':'Bundled-image software diagnostic, not a held-out experiment or physical trait validation',
            'split':'unknown','matching_iou':.5,'boundary_tolerance_original_px':2,
            'mask_convention':'GT rounded LabelMe polygons; prediction nearest-neighbour raster masks; crops inverse-rotated into original coordinates',
            'yolo_comparison_scope':'Largest cleaned raw YOLO chili candidate on the SAME threshold-run oriented crop; not a separate end-to-end YOLO-mask pipeline run',
            'color_strata':'unknown; no maturity/color metadata inferred','images':[], 'skipped':[]}
    all_metrics={'threshold':[],'yolo':[]}
    totals={method:{'matched':0,'false_positive':0,'false_negative':0} for method in all_metrics}
    for path in paths:
        annotation=args.annotation_dir/path.with_suffix('.json').name
        dimensions=cv2.imread(str(path)).shape[:2]
        with Image.open(path) as header:
            stored_dimensions=(header.height,header.width)
        data=json.loads(annotation.read_text())
        if stored_dimensions != dimensions or dimensions != (data.get('imageHeight'),data.get('imageWidth')) or not any(shape['label'].strip().lower()=='chili' for shape in data['shapes']):
            report['skipped'].append({'image':path.name,'reason':'Stored/decoded image or annotation dimensions differ, or whole-fruit labels are absent; no implicit annotation transform applied'})
            continue
        started=time.perf_counter();fruits=run_image(path,config,models)
        gt=ground_truth(annotation,fruits.width_img,fruits.height_img)
        predictions={'threshold':[],'yolo':[]}
        for c in fruits.items:
            x1,x2,y1,y2=map(int,c.position)
            predictions['threshold'].append(Instance(x1,y1,restore(c.mask_chili,c.rotation)>0))
            candidates=c.rseg['yolo_chili_candidates']
            if candidates:
                raw=max(candidates,key=lambda item:np.count_nonzero(item['mask']))['mask']
                clean=models[1]._clean_largest_chili_mask(raw)
                if np.any(clean):predictions['yolo'].append(Instance(x1,y1,restore(clean,c.rotation)>0))
        entry={'image':path.name,'annotation':str(annotation.resolve()),'ground_truth_count':len(gt),'methods':{},'elapsed_seconds':time.perf_counter()-started}
        for method in predictions:
            metrics=match_instances(predictions[method],gt)
            entry['methods'][method]=metrics
            all_metrics[method].extend(metrics['pairs'])
            for name in totals[method]:totals[method][name]+=metrics[name]
        report['images'].append(entry);manifest['images'].append(image_diagnostics(fruits))
        print(path.name,entry['methods']['threshold']['matched'],'matched',flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
    report['summary']={method:{**totals[method],**{
        name:float(np.mean([row[name] for row in all_metrics[method]])) if all_metrics[method] else None for name in ('iou','dice','boundary_f')}} for method in all_metrics}
    args.output.write_text(json.dumps(report,indent=2))
    args.output.with_name(args.output.stem+'_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
