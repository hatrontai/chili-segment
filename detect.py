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
        contour_line = np.zeros_like(fruit_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= max_area:
                cv2.drawContours(fruit_mask, [cnt], -1, (255, 255, 255), -1)

    return fruit_mask


class DetectModel:
    def __init__(self, model_name='none', weight_path= None):
        if model_name == 'yolo':
            self.model = YOLO(weight_path)
            self.use_model = True
            print('Use model yolo for detect ech chili fruit')
        elif model_name == 'none':
            self.use_model = False
            print('Use threshold for detect each chili fruit')
        else:
            self.use_model = False
            print('please set model name is yolo')


    def model_detect(self, image):
        bboxs = self.model(image)

        return bboxs

    def handle_detect(self, image):
        ep = EP
        img_RGB = image.copy()
        img_red = img_RGB[:, :, 0]
        img_mask = get_mask(img_RGB, 22 , True, False)
        
        contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        max_area = max(contour_areas)
        fruit_mask = img_RGB[540:, :, :]
        # masks = []
        bboxs = []
        # crop_images = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 10000:
                # print(cnt)
                # masks.append(cnt)
                cv2.drawContours(fruit_mask, [cnt], -1, (255, 255, 255), 2)
                length_L1 = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, ep*length_L1, True)
                contour = [p[0].tolist() for p in cnt]
                contour_or = [[p[0],p[1]+540] for p in contour]
                x, y, w, h = cv2.boundingRect(np.array(contour_or))
                bbox = [x, y, w, h]
                bboxs.append(bbox)
                # crop_image = image[y-50:y+h+50, x-50:x+w+50]

                # crop_images.append(crop_image)
                # masks.append(contour_or)
                
        return bboxs

    
    def predict(self, path):
        image = cv2.imread(path)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_crop = image_RGB[540:, :, :]
        # print(1)
        if self.use_model:
            bboxs = self.model_detect(image_crop)
        else:
            bboxs = self.handle_detect(image_crop)
        
        return bboxs