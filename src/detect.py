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
    
def get_mask(img,threshold = 16, use_closing = True, use_contour = True):
    img = img
    
    img_red = np.array(img[:, :, 0])
    hist = cv2.calcHist([img_red], [0], None, [256], [0, 256])

    fruit_mask = hadle_segm(img_red, hist.argmax() + threshold)
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

        self.device = device

    def model_detect(self, images):
        results = self.model.predict(
                    source=images,  
                    conf=0.9,
                    iou=0.7,
                    device=self.device,  
                    save=False,           
                    show=False
                )
        bboxs = results.bboxes
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
            if area >= 10000:
                cv2.drawContours(fruit_mask, [cnt], -1, (255, 255, 255), 2)
                length_L1 = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, ep*length_L1, True)
                contour = [p[0].tolist() for p in cnt]
                contour_or = [[p[0],p[1]+540] for p in contour]
                x, y, w, h = cv2.boundingRect(np.array(contour_or))
                bbox = [x, y, w, h]
                x1, y1, x2, y2 = max(0,x-50), max(0, y-50), min(W,x+w+50),min(H,y+h+50)
                if check_object(img_RGB[y1:y2, x1:x2]):
                    bboxs.append(bbox)
                
        return fruit_mask, bboxs

    
    def predict(self, path):
        image = cv2.imread(path)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_crop = image_RGB[540:, :, :]
        # print(1)
        if self.use_model:
            bboxs = self.model_detect(image_crop)
        else:
            _, bboxs = self.handle_detect(image_crop)
        
        bboxs = self.sort_bboxes_by_variance(bboxs)
        return bboxs
    
    def sort_bboxes_by_variance(self, bboxs):
        x_coords = [bbox[0] for bbox in bboxs]
        y_coords = [bbox[1] for bbox in bboxs]
        var_x = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 0
        var_y = max(x_coords) - min(y_coords) if len(y_coords) > 1 else 0
        
        if var_x > var_y:
            bboxs.sort(key=lambda b: b[0])
        else:
            bboxs.sort(key=lambda b: b[1])
            
        return bboxs