from PIL import Image
import cv2
import numpy as np

from ultralytics import YOLO


class DetectModel:
    def __init__(self, weight_path= None, model_name='none'):
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
        

        return 0
    
    def predict(self, path):
        image = cv2.im_read(path)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_crop = image_RGB[500:, :, :]

        if self.use_model:
            bboxs = self.model_detect(image_crop)
        else:
            bboxs = self.handle_detect(image_crop)
        
        return bboxs