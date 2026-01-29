from PIL import Image
from ultralytics import YOLO



def model_segment(path):
    image = Image.open(path)
    model = YOLO("yolov8n.pt")

    result = model(image)
    r = result[0]
    
    bboxs = r.boxes     # bounding boxes
    polygons = r.masks     # segmentation masks
    cls = r.probs     # classification (nếu có)
    cls_name = r.names     # mapping class_id → tên class

    polygons = modify_output(polygons)

    return polygons



def modify_output(polygons):

    return polygons