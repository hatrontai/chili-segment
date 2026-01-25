from PIL import Image


def model_detete(path):
    image = Image.open(path)

    bboxs = predict(image)
    return bboxs

def handle_detete(path):
    
    return bboxs