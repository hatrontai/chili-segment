from PIL import Image

from ultralytics import YOLO



class SegmentModel:
    def __init__(self, weight_path="./pretrain/yolov8_segment_v1.pt", model_name='yolo'):
        if model_name == 'yolo':
            self.model = YOLO(weight_path)
        elif model_name == 'rcnn':
            print('please set model name is yolo')
        else:
            print('please set model name is yolo')


    def predict(self, image):
        result = self.model(image)
        
        return result[0].masks

    def modify_output(polygons):

        return polygons
