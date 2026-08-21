from PIL import Image
import cv2
import numpy as np

from ultralytics import YOLO


EP = 0.003

def hadle_segm(img, threshold=56):
    # print(type(img))
    img = img.copy()
    img = np.array(img)
    img = np.where(img >= threshold, 255, 0)

    return img
    
def get_mask(img, threshold=16, use_closing=True, use_contour=True):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(np.uint8)

    _, fruit_mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fruit_mask = fruit_mask.astype(np.uint8)

    if use_closing:
        kernel = np.ones((10, 10), np.uint8)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        fruit_mask = fruit_mask.astype(np.uint8)
    if use_contour:
        contours, hierarchy = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        max_area = max(contour_areas)
        fruit_mask = np.zeros_like(fruit_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= max_area:
                cv2.drawContours(fruit_mask, [cnt], -1, (255, 255, 255), -1)

    return fruit_mask

def check_object(img):
    mask = get_mask(img, use_closing=False, use_contour=False)

    if mask.sum() > 0:
        return True
    else:
        return False

class DetectModel:
    def __init__(self, model_name='none', weight_path= '../pretrain/yolo26n_detection_v1.pt', device='cpu'):
        if model_name == 'yolo':
            self.model = YOLO(weight_path, task="detect")
            self.use_model = True
            print('Use model yolo for detect ech chili fruit')
        elif model_name == 'none':
            self.use_model = False
            print('Use threshold for detect each chili fruit')
        else:
            self.use_model = False
            print('please set model name is yolo')

        self.weight_path = weight_path   # luu lai de SAHI AutoDetectionModel dung chung 1 weight
        self.device = device
        self._sahi_model = None          # lazy-init, chi tao khi thuc su can (tranh loi neu chua cai sahi)

    # Kich thuoc tile va overlap cho SAHI - dieu chinh neu can (nen gan bang imgsz luc train model detect)
    SAHI_SLICE_HEIGHT = 640
    SAHI_SLICE_WIDTH = 640
    SAHI_OVERLAP_RATIO = 0.2
    SAHI_CONF_THRESHOLD = 0.5

    def model_detect(self, images):
                from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        if self._sahi_model is None:
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.weight_path,
                confidence_threshold=self.SAHI_CONF_THRESHOLD,
                device=self.device,
            )

        result = get_sliced_prediction(
            images,
            self._sahi_model,
            slice_height=self.SAHI_SLICE_HEIGHT,
            slice_width=self.SAHI_SLICE_WIDTH,
            overlap_height_ratio=self.SAHI_OVERLAP_RATIO,
            overlap_width_ratio=self.SAHI_OVERLAP_RATIO,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )

                bboxs = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            x = int(bbox.minx)
            y = int(bbox.miny) + getattr(self, 'crop_offset', 540)
            w = int(bbox.maxx - bbox.minx)
            h = int(bbox.maxy - bbox.miny)
            bboxs.append([x, y, w, h])

        return bboxs

    def handle_detect(self, image):
        ep = EP
        img_RGB = image.copy()
        W, H = img_RGB.shape[1], img_RGB.shape[0]
        img_mask = get_mask(img_RGB, 22 , True, False)
        
        contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fruit_mask = img_RGB
        bboxs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
                        if area >= 4000:
                                x_local, y_local, w_local, h_local = cv2.boundingRect(cnt)
                touches_crop_edge = y_local <= 5
                is_squarish = (w_local > 0) and (h_local / w_local) < 2.0
                if touches_crop_edge and is_squarish:
                    continue

                cv2.drawContours(fruit_mask, [cnt], -1, (255, 255, 255), 2)
                length_L1 = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, ep*length_L1, True)
                contour = [p[0].tolist() for p in cnt]
                contour_or = [[p[0],p[1] + getattr(self, 'crop_offset', 540)] for p in contour]
                x, y, w, h = cv2.boundingRect(np.array(contour_or))
                bbox = [x, y, w, h]
                pad = 50
                x1, y1, x2, y2 = max(0,x-pad), max(0, y-pad), min(W,x+w+pad),min(H,y+h+pad)
                if check_object(img_RGB[y1:y2, x1:x2]):
                    bboxs.append(bbox)
                
        return fruit_mask, bboxs

    
    def predict(self, path):
        image = cv2.imread(path)
        if image is None:
            return []
            
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Dynamic crop offset: only crop top 540 if it's a large scanned image
        self.crop_offset = 540 if image_RGB.shape[0] > 1000 else 0
        image_crop = image_RGB[self.crop_offset:, :, :]
        
        # print(1)
        if self.use_model:
            bboxs = self.model_detect(image_crop)
        else:
            _, bboxs = self.handle_detect(image_crop)
        
        bboxs = self.sort_bboxes_by_variance(bboxs)
        return bboxs
    
    def sort_bboxes_by_variance(self, bboxs):
        if len(bboxs) <= 1:
            return bboxs
        x_coords = [bbox[0] for bbox in bboxs]
        y_coords = [bbox[1] for bbox in bboxs]
        var_x = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 0
        var_y = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 0
        
        if var_x > var_y:
            bboxs.sort(key=lambda b: b[0])
        else:
            bboxs.sort(key=lambda b: b[1])
            
        return bboxs