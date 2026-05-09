from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from shapely.geometry import Polygon, MultiPolygon

from ultralytics import YOLO

# Define class names
CLASS_NAMES = {
    0: "seed",
    1: "outer",
    2: "placenta",
    3: "endocarp",
    4: "chili",
}

class SegmentModel:
    def __init__(self, weight_path="../pretrain/yolov8_segment_v1.pt", model_name='yolo', device='cpu'):
        if model_name == 'yolo':
            self.model = YOLO(weight_path, task='segment')
        else:
            raise ValueError("Unsupported model_name. Please set model_name to 'yolo'.")

        self.device = device

    def predict(self, image):
        
        target_class = 3
        epsilon_factor = 0.002
        min_dist = 4.0
        buffer_size = 2
        
        #Check input type
        if not isinstance(image, np.ndarray):
            raise ValueError("Wrong image type!")
        
        if len(image.shape) == 3:
            img_height, img_width = image.shape[:2]
        elif len(image.shape) == 2:
            img_height, img_width = image.shape
        else:
            raise ValueError(f"Wrong image shape: {image.shape}")
        
        #Predict using YOLO
        results = self.model.predict(
            source=image,  
            imgsz=640,
            conf=0.25,
            iou=0.7,
            device=self.device,
            retina_masks=True,   
            save=False,           
            show=False
        )
        
        r = results[0]
        boxes = r.boxes
        masks = r.masks
        
        processed_polygons = []
        masks_data = []
        detection_details = []
        
        if masks is not None and len(masks) > 0:
            #Parse masks and boxes
            for i in range(len(masks)):
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                
                mask = masks.data[i].cpu().numpy()
                
                # Resize mask
                if mask.shape != (img_height, img_width):
                    mask = cv2.resize(mask, (img_width, img_height), 
                                    interpolation=cv2.INTER_NEAREST)
                
                mask = (mask * 255).astype(np.uint8)
                
                masks_data.append({
                    'class_id': class_id,
                    'mask': mask,
                    'confidence': confidence,
                    'bbox': bbox
                })
                
                #Save detection details
                detection_details.append({
                    'detection_id': i,
                    'class_id': class_id,
                    'class_name': CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    'confidence': confidence,
                    'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                    'bbox_center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    'bbox_area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                    'mask_area': float(mask.sum() / 255)  # Số pixel trong mask
                })
            
            processed_polygons = self._process_yolo_masks(
                masks_data, 
                img_width, 
                img_height,
                target_class=target_class,
                epsilon_factor=epsilon_factor,
                min_dist=min_dist
            )
            
            # Modify output
            processed_polygons = self._modify_output(
                processed_polygons, 
                img_width, 
                img_height, 
                buffer_size=buffer_size
            )
        
        return {
            'raw_results': results,
            'polygons': processed_polygons,
            'detections': detection_details,
            'masks_data': masks_data,
            'image_size': (img_width, img_height),
            
            'stats': {
                'num_detections': len(detection_details),
                'num_polygons': len(processed_polygons),
                'classes_detected': list(set([d['class_id'] for d in detection_details])),
                'classes_detected_names': list(set([d['class_name'] for d in detection_details]))
            }
        }
    
    #Modify output polygons: denormalize and smooth
    def _modify_output(self, polygons, img_width, img_height, buffer_size=2):
        result = []
        
        for poly_data in polygons:
            class_id = poly_data['class_id']
            points_norm = poly_data['points_normalized']
            
            # Denormalize points to pixel coordinates
            points_pixel = points_norm.copy()
            points_pixel[:, 0] *= img_width
            points_pixel[:, 1] *= img_height
            
            #Buffer smoothing
            if buffer_size > 0:
                smoothed = self._smooth_polygon_with_buffer(points_pixel, buffer_size)
                if smoothed is not None:
                    points_pixel = smoothed
            
            result.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES.get(class_id, f"class_{class_id}"),
                'points': points_pixel,
                'points_normalized': points_norm
            })
        
        return result
    
    #Close openings at the top of the mask
    def _close_top_opening(self, mask):
        white_pixels = np.column_stack(np.where(mask > 0))
        if len(white_pixels) == 0:
            return mask
            
        points = white_pixels[:, [1, 0]]
        min_y = points[:, 1].min() 
        top_threshold = min_y + 100
        
        top_points = points[points[:, 1] <= top_threshold] 
        if len(top_points) < 2:  
            return mask
            
        center_x = mask.shape[1] // 2
        left_points = top_points[top_points[:, 0] < center_x]
        right_points = top_points[top_points[:, 0] >= center_x]
        
        result = mask.copy()
        
        if len(left_points) > 0:
            left_sorted = left_points[np.argsort(left_points[:, 1])]
            num_left = max(1, len(left_sorted) // 10)
            left_top_points = left_sorted[:num_left]
            left_top = left_top_points[np.argmin(left_top_points[:, 0])]
        else:
            left_top = None
        
        if len(right_points) > 0:
            right_sorted = right_points[np.argsort(right_points[:, 1])]
            num_right = max(1, len(right_sorted) // 10)
            right_top_points = right_sorted[:num_right]
            right_top = right_top_points[np.argmax(right_top_points[:, 0])]
        else:
            right_top = None
        
        if left_top is not None and right_top is not None:
            cv2.line(result, 
                     tuple(left_top.astype(int)), 
                     tuple(right_top.astype(int)), 
                     255, 
                     thickness=5)
        
        return result

    #Fill holes in mask
    def _fill_holes_only(self, mask):

        result = mask.copy()
        
        #Morphological close with small kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        result = self._close_top_opening(result)
        
        #Morphological close with large kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        #Binary fill holes
        result_binary = result > 0
        result_filled = binary_fill_holes(result_binary)
        result = (result_filled * 255).astype(np.uint8)
        
        #Morphological close again
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        #Final binary fill holes
        result_binary = result > 0
        result_filled = binary_fill_holes(result_binary)
        result = (result_filled * 255).astype(np.uint8)
        
        result = cv2.bitwise_or(mask, result)
        
        result_binary = result > 0
        result_filled = binary_fill_holes(result_binary)
        result = (result_filled * 255).astype(np.uint8)
        
        return result

    #Filter dense points in polygon
    def _filter_dense_points(self, points, min_dist=4.0):
        if len(points) < 3:
            return points
        
        filtered = [points[0]]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - filtered[-1])
            if dist > min_dist:
                filtered.append(points[i])
        
        if len(filtered) > 2:
            if np.linalg.norm(filtered[-1] - filtered[0]) < min_dist:
                filtered.pop()
        
        return np.array(filtered)

    #Approximate polygon with smoothing and filtering dense points
    def _approximate_polygon_smooth(self, points, epsilon_factor=0.01, min_dist=4.0):

        if len(points) < 3:
            return points
        
        #Reshape
        points_reshaped = points.reshape(-1, 1, 2).astype(np.int32)
        
        perimeter = cv2.arcLength(points_reshaped, True)
        epsilon = epsilon_factor * perimeter
        
        #ApproxPolyDP
        approx = cv2.approxPolyDP(points_reshaped, epsilon, True)
        approx_points = approx.reshape(-1, 2)
     
        filtered_points = self._filter_dense_points(approx_points, min_dist=min_dist)
        
        return filtered_points

    #Convert mask to polygon with normalization
    def _mask_to_polygon_normalized(self, mask, epsilon_factor=0.01, min_dist=4.0):

        smoothed_mask = cv2.GaussianBlur(mask, (9, 9), 0)
        _, smoothed_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        if len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour.reshape(-1, 2).astype(np.float32)
        
        points = self._approximate_polygon_smooth(points, epsilon_factor=epsilon_factor, min_dist=min_dist)
        
        #Normalize points
        h, w = mask.shape
        points_normalized = points.copy().astype(float)
        points_normalized[:, 0] /= w
        points_normalized[:, 1] /= h
        
        return points_normalized

    #Smooth polygon using buffer technique
    def _smooth_polygon_with_buffer(self, points, buffer_size=2, simplify_tolerance=1.0):
        try:
            poly = Polygon(points)
            
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            if buffer_size > 0:
                poly_smooth = (
                    poly
                    .buffer(buffer_size, resolution=4) 
                    .buffer(-buffer_size, resolution=4)
                )
                
                poly_smooth = poly_smooth.simplify(simplify_tolerance, preserve_topology=True)

                if poly_smooth.is_empty:
                    poly_smooth = poly
            else:
                poly_smooth = poly
            
            #Extract coordinates
            if isinstance(poly_smooth, Polygon):
                coords = np.array(poly_smooth.exterior.coords[:-1])
            elif isinstance(poly_smooth, MultiPolygon):
                if not poly_smooth.geoms:
                    return None
                largest_poly = max(poly_smooth.geoms, key=lambda p: p.area)
                coords = np.array(largest_poly.exterior.coords[:-1])
            else:
                return None
                
            return coords

        except Exception as e:
            print(f"Warning: Error smoothing polygon: {e}")
            return None

    #Process mask for class 'endocarp'
    def _process_yolo_masks(self, masks_data, img_width, img_height, target_class=3, 
                          epsilon_factor=0.002, min_dist=4.0):
        #Process masks
        target_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for mask_info in masks_data:
            if mask_info['class_id'] == target_class:
                # Merge
                target_mask = cv2.bitwise_or(target_mask, mask_info['mask'])
        
        if target_mask.sum() > 0:
            filled_mask = self._fill_holes_only(target_mask)
            target_polygon_norm = self._mask_to_polygon_normalized(
                filled_mask, 
                epsilon_factor=epsilon_factor,
                min_dist=min_dist
            )
        else:
            target_polygon_norm = None
        
        #Compile final polygons
        result_polygons = []
        target_class_added = False
        
        for mask_info in masks_data:
            if mask_info['class_id'] == target_class:
                if not target_class_added and target_polygon_norm is not None:
                    result_polygons.append({
                        'class_id': target_class,
                        'points_normalized': target_polygon_norm
                    })
                    target_class_added = True
            else:
                mask = mask_info['mask']
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                   
                    largest_contour = max(contours, key=cv2.contourArea)
                    points_pixel = largest_contour.reshape(-1, 2).astype(np.float32)
                    
                    approx_points = self._approximate_polygon_smooth(
                        points_pixel,
                        epsilon_factor=epsilon_factor,
                        min_dist=min_dist
                    )
                    
                    points_norm = approx_points.astype(float)
                    points_norm[:, 0] /= img_width
                    points_norm[:, 1] /= img_height
                    
                    result_polygons.append({
                        'class_id': mask_info['class_id'],
                        'points_normalized': points_norm
                    })
        
        return result_polygons