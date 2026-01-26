from PIL import Image
import cv2

def model_detete(path):
    image = Image.open(path)

    bboxs = model.predict(image)
    return bboxs

def handle_detete(path):
    image = cv2.im_read(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    return bboxs